import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb

from tqdm import tqdm
from .mlp import MLP

from .utils import sigmoid

from .mlp import MLP, Encoder
from tqdm import tqdm

from abc import ABC, abstractmethod

class Concept_FM(nn.Module, ABC):
    def __init__(
        self,
        config,
        _encoder: nn.Module = Encoder,
        _decoder: nn.Module = MLP,
        **kwargs,
    ):

        super(Concept_FM, self).__init__()

        self.input_dim = config.get("input_dim", 1024)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.latent_dim = config.get("latent_dim", 256)
        self.n_concepts = config.get("n_concepts", 10)
        self.n_unknown = config.get("n_unknown", 128)
        self.n_layers = config.get("n_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        self.p_uncond = config.get("p_uncond", 0.1)

        # Encoder
        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers =self.n_layers,
            dropout=self.dropout,
        )

        if "cb_layers" in config:
            cb_layers = config["cb_layers"]
        else:
            cb_layers = 1

        cb_concepts_layers = []
        cb_unk_layers = []

        for k in range(0, cb_layers - 1):
            layer_k = [
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            cb_concepts_layers += layer_k
            cb_unk_layers += layer_k

        cb_concepts_layers.append(nn.Linear(self.latent_dim, self.n_concepts))
        cb_concepts_layers.append(nn.Sigmoid())

        cb_unk_layers.append(nn.Linear(self.latent_dim, self.n_unknown))
        cb_unk_layers.append(nn.ReLU())

        self.cb_concepts_layers = nn.Sequential(*cb_concepts_layers)
        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        # --- VARIATIONAL CHANGE 1: Add layers for mu and logvar ---
        # These new layers will map the concept vector 'h' to the parameters
        # of a Gaussian distribution, which will serve as the prior for the flow.
        self.fc_mu = nn.Linear(self.n_concepts + self.n_unknown, self.input_dim)
        self.fc_logvar = nn.Linear(self.n_concepts + self.n_unknown, self.input_dim)
        # --- End of Change ---

        self._decoder = _decoder(
            x_dim=self.input_dim,
            c_dim=self.n_concepts + self.n_unknown,
            emb_dim=self.hidden_dim,
            n_layers = self.n_layers,
            dropout=self.dropout,
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.concepts_hp = config.get("concepts_hp", 1.0)
        self.orthogonality_hp = config.get("orthogonality_hp", 1.0)
        # --- VARIATIONAL CHANGE 2: Add KL loss hyperparameter ---
        self.kl_hp = config.get("kl_hp", 0.01) # Add a weight for the new KL loss term
        # --- End of Change ---

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        if config.get("use_soft_concepts", False):
            self.concept_loss = self._soft_concept_loss
            self.concept_transform = sigmoid
        else:
            self.concept_loss = self._hard_concept_loss
            self.concept_transform = sigmoid

    @property
    def has_concepts(self):
        return True

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)
    
    def get_prior_params(self, h):
        """
        Calculates mu and logvar from the combined concept vector h.
        These parameters define the N(mu, sigma^2) prior for the flow matching.
        """
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return dict(mu=mu, logvar=logvar)


    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):
        known_concepts = self.cb_concepts_layers(z)
        unknown = self.cb_unk_layers(z)

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts is None:
                input_concept = known_concepts
            else:
                input_concept = concepts

        h = torch.cat((input_concept, unknown), 1)

        return dict(
            input_concept=input_concept,
            pred_concept=known_concepts,
            unknown=unknown,
            h=h,
        )

    @torch.no_grad()
    def decode(self, h, n_steps: int = 1000, cfg_strength: float = 1.2, **kwargs):
        """
        Integrates the flow ODE forward from t=0 to t=1 using the Euler
        method, conditioned on the concept vector h.
        
        VARIATIONAL CHANGE: The starting point x_t at t=0 is now sampled
        from the learned prior N(mu, sigma^2) instead of N(0, I).
        """
        batch_size, device = h.shape[0], h.device

        # --- VARIATIONAL CHANGE 4 & REFACTOR: Use renamed function ---
        prior_params = self.get_prior_params(h)
        mu, logvar = prior_params["mu"], prior_params["logvar"]
        std = torch.exp(0.5 * logvar)
        # Sample the starting point x_0 from N(mu, sigma^2)
        x_t = mu + std * torch.randn_like(mu)
        # --- End of Change ---

        dt = 1.0 / n_steps

        print(f"Decoding with {n_steps} steps and cfg_strength {cfg_strength}")

        # Iteratively solve the ODE using the Euler method
        for t_step in tqdm(range(n_steps), desc="Forward Process", ncols=88, leave=False):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            v_cond = self._decoder(x_t=x_t, t=t_vec, c=h)
            v_uncond = self._decoder(x_t=x_t, t=t_vec, c=torch.zeros_like(h))
            v_guided = v_uncond + cfg_strength * (v_cond - v_uncond)

            x_t = x_t + v_guided * dt

        x_pred = x_t
        return x_pred

    # --- REFACTOR: Simplified intervene function call ---
    def intervene(self, x, concepts, mask, **kwargs):
        z = self.encode(x)
        cbm_out = self.cbm(z=z, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm_out)
        return dec
    # --- End of Change ---

    def orthogonality_loss(self, c, u):
        batch_size = u.size(0)
        u_mean = u.mean(dim=0, keepdim=True)
        c_mean = c.mean(dim=0, keepdim=True)
        u_centered = u - u_mean
        c_centered = c - c_mean
        cross_covariance = torch.matmul(u_centered.T, c_centered) / (batch_size - 1)
        loss = (cross_covariance**2).sum()
        return loss
    
    # --- VARIATIONAL CHANGE 5: New KL divergence loss function ---
    def kl_loss(self, mu, logvar):
        """
        Calculates the KL divergence between the learned distribution N(mu, sigma^2)
        and a standard normal distribution N(0, I). This regularizes the latent space.
        """
        # Sum over the dimensions and average over the batch
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()
    # --- End of Change ---

    def fm_loss(self, v_pred, v):
        return F.mse_loss(v_pred, v, reduction="mean")

    def _soft_concept_loss(self, pred_concept, concepts, **kwargs):
        return self.n_concepts * F.mse_loss(pred_concept, concepts, reduction="mean")

    def _hard_concept_loss(self, pred_concept, concepts, **kwargs):
        return self.n_concepts * F.binary_cross_entropy(pred_concept, concepts, reduction="mean")

    # --- VARIATIONAL CHANGE 6: Update main loss function ---
    def loss_function(self, v_pred, v, concepts, pred_concept, unknown, mu, logvar, **kwargs):
        """
        Calculates the total loss for the model, now including the KL divergence loss.
        """
        loss_dict = {}
        flow_loss = self.fm_loss(v_pred, v)
        loss_dict["fm_loss"] = flow_loss
        loss_dict["Total_loss"] = flow_loss

        pred_concept_clipped = torch.clip(pred_concept, 0, 1)

        if self.use_concept_loss:
            overall_concept_loss = self.concept_loss(pred_concept_clipped, concepts)
            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:
            orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
            loss_dict["orth_loss"] = orth_loss
            loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss

        # Add the new KL loss to the total loss
        kld_loss = self.kl_loss(mu, logvar)
        loss_dict["kl_loss"] = kld_loss
        loss_dict["Total_loss"] += self.kl_hp * kld_loss

        return loss_dict
    # --- End of Change ---

    def train_loop(self, data: torch.Tensor,
               concepts: torch.Tensor,
               num_epochs: int,
               batch_size: int,
               lr: float = 3e-4,
               lr_gamma: float = 0.997):
        """
        Defines the training loop for the Variational Concept_FM model.

        VARIATIONAL CHANGE: The starting point of the flow, x_0, is now sampled
        from the learned prior N(mu, sigma^2), and the KL loss is calculated.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        self.train()
        # --- 2. The Training Loop ---
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)
     
        for epoch in pbar:
            self.train()
            total_loss = 0.0
             
            for x_batch, concepts_batch in data_loader:
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)
                 
                # --- REFACTOR: Simplified encoding and CBM calls ---
                # 1. Encode the input data to a latent vector z
                z = self.encode(x_batch)

                # 2. Break z into concepts
                cbm_out = self.cbm(z=z)
                # --- End of Refactor ---
                h = cbm_out["h"]

                prior_params = self.get_prior_params(h)
                mu, logvar = prior_params["mu"], prior_params["logvar"]
                std = torch.exp(0.5 * logvar)

                # 3. Classifier-Free Guidance Training
                is_unconditional = torch.rand(x_batch.shape[0], device=device) < self.p_uncond
                h_guided = torch.where(is_unconditional.unsqueeze(-1), 0.0, h)

                # Sample x_0 from the learned distribution N(mu, sigma^2)
                x_0 = mu + std * torch.randn_like(mu)
                
                # The target x_1 remains the real data
                x_1 = x_batch

                # 5. Prepare for Flow Matching Loss
                t = torch.rand(x_batch.shape[0], 1, device=device)
                x_t = (1 - t) * x_0 + t * x_1
                v = x_1 - x_0
                # --- End of Change ---

                # 6. Get the Model's Prediction
                v_pred = self._decoder(x_t=x_t, t=t.squeeze(-1), c=h_guided)

                # 7. Calculate loss (now including mu and logvar)
                loss_dict = self.loss_function(
                    v_pred=v_pred,
                    v=v,
                    concepts=concepts_batch,
                    pred_concept=cbm_out["pred_concept"],
                    unknown=cbm_out["unknown"],
                    mu=mu,
                    logvar=logvar, # Pass new params to the loss function
                )
                loss = loss_dict["Total_loss"]

                # 8. Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                 
                total_loss += loss.item()

            # --- End of Epoch ---
            avg_loss = total_loss / len(data_loader)
             
            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
             
            scheduler.step()

        self.eval()

