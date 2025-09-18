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
from .decoder import DecoderBlock, SkipDecoderBlock

EPS = 1e-6




class CB_VAE_MIXED(BaseCBVAE):
    def __init__(
    self,
    config,
    _encoder: nn.Module = EncoderBlock,
    _decoder: nn.Module = SkipDecoderBlock,
    **kwargs,
    ):
        # --- MODIFICATION: Handle different config formats for reverse compatibility ---
        # Check if the new, more explicit config keys are present.
        if "use_hard_concepts" in config or "use_soft_concepts" in config or \
           "n_hard_concepts" in config or "n_soft_concepts" in config:
            # New format logic
            self.use_hard_concepts = config.get("use_hard_concepts", False)
            self.use_soft_concepts = config.get("use_soft_concepts", False)
            self.n_hard_concepts = config.get("n_hard_concepts", 0) if self.use_hard_concepts else 0
            self.n_soft_concepts = config.get("n_soft_concepts", 0) if self.use_soft_concepts else 0
        else:
            # Old format logic: infer from 'n_concepts' and 'use_soft_concepts'
            n_concepts_total = config.get("n_concepts", 0)
            is_soft = config.get("use_soft_concepts", False) # Default to hard concepts

            if is_soft:
                self.use_hard_concepts = False
                self.n_hard_concepts = 0
                self.use_soft_concepts = True
                self.n_soft_concepts = n_concepts_total
            else: # is_hard
                self.use_hard_concepts = True
                self.n_hard_concepts = n_concepts_total
                self.use_soft_concepts = False
                self.n_soft_concepts = 0

        # Update config with total number of concepts for the base class
        config["n_concepts"] = self.n_hard_concepts + self.n_soft_concepts
        if config["n_concepts"] == 0:
            raise ValueError("The model must have at least one hard or soft concept.")
        
        super().__init__(
            config,
            **kwargs,
        )

        # if decoder is SkipDecoderBlock print message
        if _decoder == SkipDecoderBlock:
            print("Using Skip Decoder")
        else:
            print("Using Residual Decoder")

        self.dropout = config.get("dropout", 0.0)
        self.encoder = _encoder(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, n_layers=self.n_layers,
            latent_dim=self.latent_dim, dropout=self.dropout
        )
        
        n_unknown = config.get("n_unknown", 32)
        cb_layers_depth = config.get("cb_layers", 1)

        self.cb_hard_layers = self._create_cb_projection(
            depth=cb_layers_depth, output_dim=self.n_hard_concepts, final_activation=nn.Sigmoid()
        ) if self.use_hard_concepts and self.n_hard_concepts > 0 else None

        self.cb_soft_layers = self._create_cb_projection(
            depth=cb_layers_depth, output_dim=self.n_soft_concepts, final_activation=False
        ) if self.use_soft_concepts and self.n_soft_concepts > 0 else None

        self.cb_unk_layers = self._create_cb_projection(
            depth=cb_layers_depth, output_dim=n_unknown, final_activation=nn.ReLU()
        )

        self.decoder = _decoder(
            input_dim=self.input_dim, n_layers=self.n_layers, n_concepts=self.n_concepts,
            n_unknown=n_unknown, hidden_dim=self.hidden_dim, dropout=self.dropout
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp
        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)
        self.concept_loss_fn = self._mixed_concept_loss



    @property
    def has_concepts(
        self,
    ):
        return True

    def _extra_loss(self, loss_dict, *args, **kwargs):
        return loss_dict

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, input_concept, unknown, **kwargs):
        return self.decoder(input_concept, unknown, **kwargs)

    def _create_cb_projection(self, depth, output_dim, final_activation):
        """Helper function to create a projection network."""
        layers = []
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), nn.Dropout(p=self.dropout),
            ])
        layers.append(nn.Linear(self.latent_dim, output_dim))
        if final_activation: layers.append(final_activation)
        return nn.Sequential(*layers)


    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):
        pred_concepts_parts = []
        if self.cb_hard_layers: pred_concepts_parts.append(self.cb_hard_layers(z))
        if self.cb_soft_layers: pred_concepts_parts.append(self.cb_soft_layers(z))
        
        known_concepts = torch.cat(pred_concepts_parts, dim=1)
        unknown = self.cb_unk_layers(z)

        input_concept = known_concepts * (1 - mask) + concepts * mask if intervene else \
                        concepts if concepts is not None else known_concepts

        h = torch.cat((input_concept, unknown), 1)
        return dict(input_concept=input_concept, 
                    pred_concept=known_concepts, 
                    unknown=unknown, 
                    h=h)
    
    def _mixed_concept_loss(self, pred_concept, concepts, **kwargs):
        total_concept_loss = 0.0
        
        if self.use_hard_concepts and self.n_hard_concepts > 0:
            pred_hard = pred_concept[:, :self.n_hard_concepts]
            concepts_hard = concepts[:, :self.n_hard_concepts]
            hard_loss = F.binary_cross_entropy(pred_hard, concepts_hard, reduction="mean")
            total_concept_loss += self.n_hard_concepts * hard_loss

        if self.use_soft_concepts and self.n_soft_concepts > 0:
            pred_soft = pred_concept[:, self.n_hard_concepts:]
            concepts_soft = concepts[:, self.n_hard_concepts:]
            soft_loss = F.mse_loss(pred_soft, concepts_soft, reduction="mean")
            total_concept_loss += self.n_soft_concepts * soft_loss
            
        return total_concept_loss

    def forward(self, x, concepts=None, **kwargs):
        enc = self.encode(x, concepts=concepts, **kwargs)
        z_dict = self.reparametrize(**enc)
        cbm_dict = self.cbm(**z_dict, concepts=concepts, **enc)
        dec_dict = self.decode(**enc, **z_dict, **cbm_dict, concepts=concepts)

        out = {}
        for d in [enc, z_dict, cbm_dict, dec_dict]:
            out.update(d)
        return out
    
    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm)
        return dec

    def orthogonality_loss(self, c, u):

        batch_size = u.size(0)

        # Compute means along the batch dimension
        u_mean = u.mean(dim=0, keepdim=True)
        c_mean = c.mean(dim=0, keepdim=True)

        # Center the variables in-place to save memory
        u_centered = u - u_mean
        c_centered = c - c_mean

        # Compute the cross-covariance matrix using batch dimension
        cross_covariance = torch.matmul(u_centered.T, c_centered) / (batch_size - 1)

        # Frobenius norm squared of the cross-covariance matrix
        loss = (cross_covariance**2).sum()

        return loss

    def rec_loss(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction="mean")


    def loss_function(self, x, concepts, x_pred, mu, logvar, pred_concept, unknown, **kwargs):
        loss_dict = {}
        MSE, KLD = self.rec_loss(x_pred, x), self.KL_loss(mu, logvar)
        loss_dict["rec_loss"], loss_dict["KL_loss"] = MSE, KLD
        loss_dict["Total_loss"] = MSE + self.beta * KLD
        pred_concept_clipped = t.clip(pred_concept, 0, 1)
        if self.use_concept_loss:
            concept_loss = self.concept_loss_fn(pred_concept_clipped, concepts)
            loss_dict["concept_loss"] = concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * concept_loss
        if self.use_orthogonality_loss:
            orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
            loss_dict["orth_loss"] = orth_loss
            loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss
        return self._extra_loss(loss_dict, x=x, concepts=concepts, x_pred=x_pred, mu=mu, logvar=logvar, pred_concept=pred_concept, unknown=unknown, **kwargs)


        

    def train_loop(self, data: torch.Tensor, 
                          hard_concepts: torch.Tensor = None,
                          soft_concepts: torch.Tensor = None, 
                          concepts: torch.Tensor = None,
                          num_epochs: int = 100, 
                          batch_size: int = 64, 
                          lr: float = 3e-4, 
                          lr_gamma: float = 0.997):
            
        if concepts is not None:
            if self.use_hard_concepts and not self.use_soft_concepts:
                hard_concepts = concepts
            elif self.use_soft_concepts and not self.use_hard_concepts:
                soft_concepts = concepts
            else:
                raise ValueError("Ambiguous 'concepts' input. Provide 'hard_concepts' and/or 'soft_concepts'.")
        
        if (hard_concepts is not None) != self.use_hard_concepts:
            raise ValueError("Mismatch between provided 'hard_concepts' and model config.")
        if (soft_concepts is not None) != self.use_soft_concepts:
            raise ValueError("Mismatch between provided 'soft_concepts' and model config.")

        torch.set_flush_denormal(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset_tensors = [data]
        if self.use_hard_concepts: dataset_tensors.append(hard_concepts)
        if self.use_soft_concepts: dataset_tensors.append(soft_concepts)
        
        dataset = TensorDataset(*dataset_tensors)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")
        self.train()
        
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=150)
        for epoch in pbar:
            total_loss, epoch_tp, epoch_fp, epoch_fn, epoch_soft_mse = 0.0, 0.0, 0.0, 0.0, 0.0
            
            for batch in data_loader:
                x_batch = batch[0].to(device, non_blocking=True)
                
                concept_parts = []
                tensor_idx = 1
                if self.use_hard_concepts:
                    concept_parts.append(batch[tensor_idx])
                    tensor_idx += 1
                if self.use_soft_concepts:
                    concept_parts.append(batch[tensor_idx])
                concepts_batch = torch.cat(concept_parts, dim=1).to(device, non_blocking=True)

                out = self.forward(x_batch)
                loss_dict = self.loss_function(x_batch, concepts_batch, **out)
                loss = loss_dict["Total_loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                with torch.no_grad():
                    if self.use_hard_concepts and self.n_hard_concepts > 0:
                        pred_concepts_hard = out["pred_concept"][:, :self.n_hard_concepts]
                        true_labels_hard = concepts_batch[:, :self.n_hard_concepts]
                        predicted_labels = (pred_concepts_hard > 0.5).float()
                        epoch_tp += (predicted_labels * true_labels_hard).sum().item()
                        epoch_fp += (predicted_labels * (1 - true_labels_hard)).sum().item()
                        epoch_fn += ((1 - predicted_labels) * true_labels_hard).sum().item()
                    
                    if self.use_soft_concepts and self.n_soft_concepts > 0:
                        pred_soft = out["pred_concept"][:, self.n_hard_concepts:]
                        true_soft = concepts_batch[:, self.n_hard_concepts:]
                        epoch_soft_mse += F.mse_loss(pred_soft, true_soft, reduction='sum').item()
            
            avg_loss = total_loss / len(data_loader)
            epsilon = 1e-7
            precision = epoch_tp / (epoch_tp + epoch_fp + epsilon)
            recall = epoch_tp / (epoch_tp + epoch_fn + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon) if self.use_hard_concepts else 0.0
            avg_soft_mse = (epoch_soft_mse / (len(data_loader.dataset) * self.n_soft_concepts)) if self.use_soft_concepts and self.n_soft_concepts > 0 else 0.0

            pbar_postfix = {
                "avg_loss": f"{avg_loss:.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.5e}"
            }
            if self.use_hard_concepts:
                pbar_postfix["hard_concept_f1"] = f"{f1_score:.4f}"
            if self.use_soft_concepts:
                pbar_postfix["soft_concept_mse"] = f"{avg_soft_mse:.4f}"
            
            pbar.set_postfix(pbar_postfix)
            scheduler.step()
        print("Training finished.")
        self.eval()