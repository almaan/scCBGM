import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb

from tqdm import tqdm

from .base import BaseCBVAE
from .utils import sigmoid

from .encoder import EncoderBlock
from .decoder import FlowDecoder

EPS = 1e-6




class CB_VAE_FM(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = EncoderBlock,
        _decoder: nn.Module = FlowDecoder,
        **kwargs,
    ):
        super().__init__(
            config,
            **kwargs,
        )

        self.dropout = config.get("dropout", 0.0)

        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        n_unknown = config.get("n_unknown", 32)
        cb_layers = config.get("cb_layers", 1)

        # Concept bottleneck layers
        cb_concepts_layers = []
        cb_unk_layers = []
        for _ in range(cb_layers - 1):
            layer_k = [nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), nn.Dropout(p=self.dropout)]
            cb_concepts_layers.extend(layer_k)
            cb_unk_layers.extend(layer_k)

        cb_concepts_layers.extend([nn.Linear(self.latent_dim, self.n_concepts), nn.Sigmoid()])
        cb_unk_layers.extend([nn.Linear(self.latent_dim, n_unknown), nn.ReLU()])
        
        self.cb_concepts_layers = nn.Sequential(*cb_concepts_layers)
        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        # --- Initialize the FlowDecoder ---
        self.bottleneck_dim = self.n_concepts + n_unknown
        
        self._decoder = _decoder(
            x_dim=self.input_dim,
            c_dim=self.bottleneck_dim,
            emb_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
        
        # Hyperparameters
        self.dropout = nn.Dropout(p=self.dropout)

        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp

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
    
    def reparametrize(self, mu, logvar, **kwargs):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return dict(z=mu + eps * std)

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)

    def cbm(self, z, **kwargs):
        known_concepts = self.cb_concepts_layers(z)
        unknown = self.cb_unk_layers(z)
        h = torch.cat((known_concepts, unknown), 1)
        return dict(pred_concept=known_concepts, unknown=unknown, h=h)

    def decode(self, h, **kwargs):
        # For inference, we integrate
        return self._decoder.integrate(h)

    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z_dict = self.reparametrize(**enc)
        cbm_dict = self.cbm(**z_dict, concepts=concepts, mask=mask, intervene=True)
        dec_dict = self.decode(**cbm_dict)
        return dec_dict


    def forward(self, x, **kwargs):
        # Forward pass for training
        enc = self.encode(x, **kwargs)
        z_dict = self.reparametrize(**enc)
        cbm_dict = self.cbm(**z_dict)
        
        out = {}
        for d in [enc, z_dict, cbm_dict]:
            out.update(d)
        return out
        
    def orthogonality_loss(self, c, u):
        batch_size = u.size(0)
        u_mean = u.mean(dim=0, keepdim=True)
        c_mean = c.mean(dim=0, keepdim=True)
        u_centered = u - u_mean
        c_centered = c - c_mean
        cross_covariance = torch.matmul(u_centered.T, c_centered) / (batch_size - 1)
        return (cross_covariance**2).sum()
        
    def fm_loss(self, v_pred, v):
        return F.mse_loss(v_pred, v, reduction="mean")

    def _soft_concept_loss(self, pred_concept, concepts, **kwargs):
        overall_concept_loss = self.n_concepts * F.mse_loss(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

    def _hard_concept_loss(self, pred_concept, concepts, **kwargs):
        overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

        
    def loss_function(self, x, concepts, h, mu, logvar, pred_concept, unknown, **kwargs):
        loss_dict = {}

        # --- Flow Matching Loss Calculation ---
        x1 = x # Target is the input data
        x0 = self._decoder.get_x0(h)
        
        # 1. Sample time t
        t = torch.rand(x.size(0), 1, device=x.device)
        
        # 2. Form the paths and target velocity
        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0 # Target velocity vector field
        
        # 3. Predict velocity
        pred_v = self._decoder(xt, t, h)
        
        # 4. Calculate FM loss (MSE)
        fm_loss = self.fm_loss(pred_v, ut)
        loss_dict["fm_loss"] = fm_loss
        loss_dict["Total_loss"] = fm_loss
        # --- End of Flow Matching Loss ---
        
        KLD = self.KL_loss(mu, logvar)
        loss_dict["KL_loss"] = KLD
        loss_dict["Total_loss"] += self.beta * KLD
        
        pred_concept_clipped = torch.clamp(pred_concept, 0, 1)
        
        overall_concept_loss = self.concept_loss(pred_concept_clipped, concepts)
        loss_dict["concept_loss"] = overall_concept_loss
        loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss
        
        orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
        loss_dict["orth_loss"] = orth_loss
        loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss
        
        return loss_dict

    def train_loop(self, data, concepts, num_epochs, batch_size, lr=3e-4, lr_gamma=0.997):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")
        self.train()
        
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=150)
        for epoch in pbar:
            total_loss = 0.0
            total_fm_loss = 0.0
            
            for x_batch, concepts_batch in data_loader:
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)
                
                # 1. Forward pass to get latent space and bottleneck
                out = self.forward(x_batch)
                
                # 2. Calculate loss
                loss_dict = self.loss_function(x_batch, concepts_batch, **out)
                loss = loss_dict["Total_loss"]
                
                # 3. Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_fm_loss += loss_dict["fm_loss"].item()
            
            avg_loss = total_loss / len(data_loader)
            avg_fm_loss = total_fm_loss / len(data_loader)
            
            pbar.set_postfix({
                "avg_total_loss": f"{avg_loss:.3e}",
                "avg_fm_loss": f"{avg_fm_loss:.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.5e}"
            })
            scheduler.step()
        
        print("Training finished.")
        self.eval()
