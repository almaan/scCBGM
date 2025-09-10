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


from .utils import sigmoid

from .mlp import FlowDecoder, DenseEncoder, ConditionEmbedding
from tqdm import tqdm

from abc import ABC, abstractmethod


class Concept_FM(nn.Module, ABC):
    def __init__(
        self,
        config,
        _encoder: nn.Module = DenseEncoder,
        _decoder: nn.Module = FlowDecoder,
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
        self.unknown_activation = config.get("unknown_activation", "hypersphere")

        if(self.p_uncond > 0):
            self.condition_embedder = ConditionEmbedding(c_dim=self.n_concepts + self.n_unknown, emb_dim=self.hidden_dim)
            self.concept_emb_dim = self.hidden_dim
        else:
            self.condition_embedder = nn.Identity()
            self.concept_emb_dim = self.n_concepts + self.n_unknown

        # Encoder
        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers =self.n_layers,
            dropout=self.dropout,
        )

        self.cb_layers = config.get("cb_layers", 2)


        self.cb_concepts_layers = _encoder(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_concepts,
            n_layers=self.cb_layers,
            dropout=self.dropout,
            output_activation='sigmoid')

        self.cb_unk_layers = _encoder(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.n_unknown,
            n_layers=self.cb_layers,
            dropout=self.dropout,
            output_activation='hypersphere' if self.unknown_activation == 'hypersphere' else 'relu')

        # --- VARIATIONAL CHANGE 1: Add layers for mu and logvar ---
        # These new layers will map the concept vector 'h' to the parameters
        # of a Gaussian distribution, which will serve as the prior for the flow.

        self.fc_mu_logvar = _encoder(
            input_dim=self.n_concepts + self.n_unknown,
            hidden_dim=self.hidden_dim,
            latent_dim=self.input_dim*2,
            n_layers =2,
            dropout=self.dropout,
        )
    
        # --- End of Change ---

        self._decoder = _decoder(
            x_dim=self.input_dim,
            c_dim=self.concept_emb_dim,
            emb_dim=self.hidden_dim,
            n_layers = self.n_layers,
            dropout=self.dropout,
        )


        self.flow_hp = config.get("flow_hp", 1.0)
        self.concepts_hp = config.get("concepts_hp", 0.1)
        self.orthogonality_hp = config.get("orthogonality_hp", 0.2)
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

        # print(self)

    @property   
    def has_concepts(self):
        return True

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)
    
    def get_prior_params(self, h):
        mu_logvar = self.fc_mu_logvar(h)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return dict(mu=mu, logvar=logvar)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):
        known_concepts = self.cb_concepts_layers(z)

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts is None:
                input_concept = known_concepts
            else:
                input_concept = concepts

        output_dict = {}
        if self.n_unknown > 0 and self.cb_unk_layers is not None:
            unknown = self.cb_unk_layers(z)
            h = torch.cat((input_concept, unknown), 1)
            output_dict["unknown"] = unknown
        else:
            h = input_concept

        output_dict.update({
            "input_concept": input_concept,
            "pred_concept": known_concepts,
            "h": h
        })
        return output_dict


    @torch.no_grad()
    def edit(self, 
             x: torch.Tensor, 
             c: torch.Tensor, 
             c_prime: torch.Tensor, 
             t_edit: float, 
             n_steps: int = 1000, 
             w_cfg_forward: float = 1.0, 
             w_cfg_backward: float = 1.0,
             noise_add: float = 0.1):
        """
        Edits a sample 'x' from its original condition 'c' to a new condition 'c_prime'.
        
        This is done by first integrating backwards from x (at t=1) to an intermediate
        time t_edit using the original condition c. Then, it integrates forward from
        that point to t=1 using the new condition c_prime.

        Args:
            x (torch.Tensor): The initial data sample to edit.
            c (torch.Tensor): The original condition vector for x.
            c_prime (torch.Tensor): The target condition vector for editing.
            t_edit (float): The time to integrate back to (between 0.0 and 1.0).
            n_steps (int): Total number of integration steps for a full t=0 to t=1 pass.
            w_cfg_forward (float): The CFG scale for the forward (editing) pass.
            w_cfg_backward (float): The CFG scale for the backward (inversion) pass.
        """
        batch_size, device = x.shape[0], x.device
        dt = 1.0 / n_steps
        
        # --- Snap t_edit to the discrete time grid ---
        n_edit_steps = int(t_edit * n_steps)
        print(f"Editing from t=1.0 back to t={n_edit_steps * dt:.2f}, then forward with new condition.")
        
        x_t = x.clone()

        c_emb = self.condition_embedder(c)
        c_prime_emb = self.condition_embedder(c_prime)
        # --- 1. Backward Process (Inversion) ---
        # Integrate from t=1 down to t_edit using the original condition 'c'
        for t_step in tqdm(range(n_steps, n_edit_steps, -1), desc="Backward Process (Inversion), cfg: {:.2e}".format(w_cfg_backward), ncols=88, leave=False):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)
            
            # Use the original condition 'c' and backward cfg scale
            v_cond = self._decoder(x_t=x_t, t=t_vec, c=c_emb)
            v_uncond = self._decoder(x_t=x_t, t=t_vec, c= torch.zeros_like(c_emb, device=device))

            v_guided = v_uncond + w_cfg_backward * (v_cond - v_uncond)

            
            # To go backward in time, we subtract the vector field
            x_t = x_t - v_guided * dt

        if(noise_add > 0):
            print("Adding noise to the edited sample with std = {:.2e}".format(noise_add))
        
        x_t = x_t + noise_add * torch.randn_like(x_t)
        # --- 2. Forward Process (Editing) ---
        # Integrate from t_edit up to t=1 using the new condition 'c_prime'
        for t_step in tqdm(range(n_edit_steps, n_steps), desc="Forward Process (Editing), cfg: {:.2e}".format(w_cfg_forward), ncols=88, leave=False):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            # Use the new condition 'c_prime' and forward cfg scale
            v_cond = self._decoder(x_t=x_t, t=t_vec, c=c_prime_emb)
            v_uncond = self._decoder(x_t=x_t, t=t_vec, c=torch.zeros_like(c_prime_emb, device=device))
            v_guided = v_uncond + w_cfg_forward * (v_cond - v_uncond)


            # To go forward in time, we add the vector field
            x_t = x_t + v_guided * dt
            
        x_edited = x_t
        return x_edited
    
    @torch.no_grad()
    def decode(self, h, n_steps: int = 1000, cfg_strength: float = 1.0, **kwargs):
        batch_size, device = h.shape[0], h.device

        prior_params = self.get_prior_params(h)
        mu, logvar = prior_params["mu"], prior_params["logvar"]
        std = torch.exp(0.5 * logvar)
        x_t = mu + std * torch.randn_like(mu)
        # x_t = torch.randn(batch_size, self.input_dim, device=device)
        
        dt = 1.0 / n_steps

        print(f"Decoding with {n_steps} steps and cfg_strength {cfg_strength}")

        h_emb = self.condition_embedder(h)
        for t_step in tqdm(range(n_steps), desc="Forward Process", ncols=88, leave=False):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            v_cond = self._decoder(x_t=x_t, t=t_vec, c=h_emb)
            v_uncond = self._decoder(x_t=x_t, t=t_vec, c=torch.zeros_like(h_emb))
            v_guided = v_uncond + cfg_strength * (v_cond - v_uncond)

            x_t = x_t + v_guided * dt

        x_pred = x_t
        return x_pred

    def intervene(self, x, concepts, mask, **kwargs):
        z = self.encode(x)
        cbm_out = self.cbm(z=z, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm_out)
        return dec

    def orthogonality_loss(self, c, u):
        batch_size = u.size(0)
        # u_mean = u.mean(dim=0, keepdim=True)
        # c_mean = c.mean(dim=0, keepdim=True)
        c_centered = (c - self.known_concept_mean) / self.known_concept_std

        if(self.unknown_activation == 'relu'):
            u_mean = u.mean(dim=0, keepdim=True)
            u_centered = u - u_mean
        else:
            u_mean = u.mean(dim=0, keepdim=True)
            u_centered = u - u_mean

        cross_covariance = torch.matmul(u_centered.T, c_centered) / (batch_size - 1)
        loss = (cross_covariance**2).sum()
        return loss
    
    def kl_loss(self, mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    # --- End of Change ---

    def fm_loss(self, v_pred, v):
        return F.mse_loss(v_pred, v, reduction="mean")

    def _soft_concept_loss(self, pred_concept, concepts, **kwargs):
        return self.n_concepts * F.mse_loss(pred_concept, concepts, reduction="mean")

    def _hard_concept_loss(self, pred_concept, concepts, **kwargs):
        return self.n_concepts * F.binary_cross_entropy(pred_concept, concepts, reduction="mean")

    def loss_function(self, v_pred, v, concepts, pred_concept, mu, logvar, unknown=None, **kwargs):
        loss_dict = {}
        flow_loss = self.fm_loss(v_pred, v)
        
        loss_dict["fm_loss"] = flow_loss

        loss_dict["Total_loss"] = 0
        loss_dict["Total_loss"] += self.flow_hp * flow_loss

        pred_concept_clipped = torch.clip(pred_concept, 0, 1)

        if self.use_concept_loss:
            overall_concept_loss = self.concept_loss(pred_concept_clipped, concepts)
            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:
            if unknown is not None:
                orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
                loss_dict["orth_loss"] = orth_loss
                loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss
            else:
                print("Warning: use_orthogonality_loss is True, but 'unknown' tensor was not provided.")

        kld_loss = self.kl_loss(mu, logvar)
        loss_dict["kl_loss"] = kld_loss
        loss_dict["Total_loss"] += self.kl_hp * kld_loss

        return loss_dict

    def train_loop(self, data: torch.Tensor,
               concepts: torch.Tensor,
               num_epochs: int,
               batch_size: int,
               lr: float = 3e-4,
               lr_gamma: float = 0.997):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(device)

        self.known_concept_mean = concepts.mean(dim=0).to('cuda')
        self.known_concept_std = concepts.std(dim=0).to('cuda') + 1e-6

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        self.train()
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=150)

        self.losses = []

        for epoch in pbar:
            self.train()
            total_loss = 0.0
            total_fm_loss = 0.0
            
            # --- CHANGE: Initialize counters for F1 score calculation ---
            # We accumulate TP, FP, FN over the epoch for a more stable F1 score
            epoch_tp = 0.0
            epoch_fp = 0.0
            epoch_fn = 0.0
            # --- End of Change ---
             
            for x_batch, concepts_batch in data_loader:
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)
                 
                z = self.encode(x_batch)
                cbm_out = self.cbm(z=z)
                h = cbm_out["h"]

                prior_params = self.get_prior_params(h)
                mu, logvar = prior_params["mu"], prior_params["logvar"]
                std = torch.exp(0.5 * logvar)

                h_emb = self.condition_embedder(h)
                is_unconditional = torch.rand(x_batch.shape[0], device=device) < self.p_uncond
                h_guided = torch.where(is_unconditional.unsqueeze(-1), 0.0, h_emb)

                x_0 = mu + std * torch.randn_like(mu)
                x_1 = x_batch

                t = torch.rand(x_batch.shape[0], 1, device=device)
                x_t = (1 - t) * x_0 + t * x_1
                v = x_1 - x_0

                v_pred = self._decoder(x_t=x_t, t=t.squeeze(-1), c=h_guided)
                
                unknown_tensor = cbm_out.get("unknown", None)
                
                loss_dict = self.loss_function(
                    v_pred=v_pred,
                    v=v,
                    concepts=concepts_batch,
                    pred_concept=cbm_out["pred_concept"],
                    unknown=unknown_tensor,
                    mu=mu,
                    logvar=logvar,
                )
                loss = loss_dict["Total_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                 
                total_loss += loss.item()
                total_fm_loss += loss_dict["fm_loss"].item()
                
                # --- CHANGE: Calculate and accumulate TP, FP, FN for the batch ---
                with torch.no_grad():
                    pred_concepts = cbm_out["pred_concept"]
                    predicted_labels = (pred_concepts > 0.5).float()
                    true_labels = concepts_batch
                    
                    # True Positives: predicted is 1 and true is 1
                    epoch_tp += (predicted_labels * true_labels).sum().item()
                    # False Positives: predicted is 1 and true is 0
                    epoch_fp += (predicted_labels * (1 - true_labels)).sum().item()
                    # False Negatives: predicted is 0 and true is 1
                    epoch_fn += ((1 - predicted_labels) * true_labels).sum().item()
                # --- End of Change ---

            # --- CHANGE: Calculate epoch-level F1 score and update progress bar ---
            avg_loss = total_loss / len(data_loader)
            avg_fm_loss = total_fm_loss / len(data_loader)
            
            # Calculate precision, recall, and F1
            epsilon = 1e-7 # To avoid division by zero
            precision = epoch_tp / (epoch_tp + epoch_fp + epsilon)
            recall = epoch_tp / (epoch_tp + epoch_fn + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
            
            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.3e}",
                "fm_loss": f"{avg_fm_loss:.3e}",
                "concept_f1": f"{f1_score:.3e}", # Display F1 score
            })
            # --- End of Change ---
            self.losses.append(avg_loss)
             
             # Log to wandb
            scheduler.step()

        self.eval()