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

# We assume BaseCBVAE provides the pl.LightningModule base,
# self.reparametrize(), and self.KL_loss()
from .base import BaseCBVAE
from .utils import sigmoid

# --- NEW: Import the provided blocks ---
from .encoder import CVAEEncoderBlock
from .decoder import CVAEDecoderBlock

EPS = 1e-6


class scCVAE(BaseCBVAE):
    """
    A Conditional Variational Autoencoder (CVAE).

    This model learns to reconstruct 'x' given a conditional input 'concepts'.
    The 'concepts' vector is fed into both the encoder and the decoder.

    This class inherits from BaseCBVAE to reuse the VAE boilerplate like
    reparametrize() and KL_loss().
    """
    def __init__(
        self,
        config,
        **kwargs,
    ):

        super().__init__(
            config,
            **kwargs,
        )

        self.dropout = config.get("dropout", 0.0)
        self.n_concepts = config.n_concepts # Get n_concepts from config
        self.beta = config.beta

        # --- 1. Define Encoder ---
        # MODIFIED: Use CVAEEncoderBlock
        # We now pass the original 'input_dim' of x, not the concatenated one.
        # We assume CVAEEncoderBlock will handle concatenation internally.
        self.encoder = CVAEEncoderBlock(
            input_dim=self.input_dim,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            n_concepts=self.n_concepts
        )

        # --- 2. Define Decoder ---
        # MODIFIED: Use CVAEDecoderBlock
        # Its 'n_latent' argument corresponds to our 'latent_dim'
        self.decoder = CVAEDecoderBlock(
            input_dim=self.input_dim,
            n_concepts=self.n_concepts,
            n_latent=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

        self.dropout = nn.Dropout(p=self.dropout)


    def cbm(self, z, **kwargs):
        """
        Dummy implementation to satisfy the abstract method requirement
        from BaseCBVAE. This model does not use a CBM.
        """
        # This method is unused in the CVAE workflow.
        # We return an empty dict to match the expected output type.
        return {}
    
    @property
    def has_concepts(
        self,
    ):
        # This model doesn't *predict* concepts, but it *uses* them.
        # Setting to False as it doesn't have a concept *bottleneck*.
        return False

    def _extra_loss(self, loss_dict, *args, **kwargs):
        # No extra losses
        return loss_dict

    def encode(self, x, concepts, **kwargs):
        """
        Encodes the input 'x' conditioned on 'concepts'.
        MODIFIED: Calls the new self.encoder module with separate arguments,
        assuming CVAEEncoderBlock.forward(x, concepts)
        """
        # Pass separate tensors to the encoder
        return self.encoder(x=x, input_concept=concepts, **kwargs)


    def decode(self, z, concepts, **kwargs):
        """
        Decodes the concatenated input (z + concepts).
        MODIFIED: Signature changed to accept z and concepts separately,
        matching the CVAEDecoderBlock's forward method.
        """
        # Pass z and concepts as separate 'latent' and 'input_concept' args
        return self.decoder(latent=z, input_concept=concepts, **kwargs)


    def forward(self, x, concepts=None, **kwargs):
        """
        Full CVAE forward pass.
        """
        if concepts is None:
            raise ValueError("CVAE requires 'concepts' (the condition) to be provided during forward pass.")

        # 1. Prepare encoder input
        # MODIFIED: No longer concatenating here.
        # enc_input = torch.cat((x, concepts), 1)
        
        # 2. Encode
        # MODIFIED: Pass 'x' and 'concepts' as separate arguments
        enc_dict = self.encode(x, concepts)
        
        # 3. Reparameterize
        z_dict = self.reparametrize(**enc_dict) # From BaseCBVAE

        # 4. Prepare decoder input
        # MODIFIED: No longer need to concatenate for the decoder
        # dec_input = torch.cat((z_dict["z"], concepts), 1)
        
        # 5. Decode
        # MODIFIED: Pass 'z' and 'concepts' as separate arguments
        dec_dict = self.decode(z=z_dict["z"], concepts=concepts)

        # 6. Collate outputs
        out = {}
        for d in [enc_dict, z_dict, dec_dict]:
            out.update(d)
        return out
    
    def intervene(self, x, concepts_enc, concepts_dec, **kwargs):
        if concepts_enc is None or concepts_dec is None:
            raise ValueError("CVAE intervention requires both 'concepts_enc' and 'concepts_dec' to be provided.")

        # 1. Encode x conditioned on 'concepts_enc'
        enc_dict = self.encode(x, concepts=concepts_enc, **kwargs)
        
        # 2. Reparameterize to get z
        z_dict = self.reparametrize(**enc_dict) # From BaseCBVAE
        
        # 3. Decode z conditioned on 'concepts_dec'
        dec_dict = self.decode(z=z_dict["z"], concepts=concepts_dec, **kwargs)

        # 4. Collate and return outputs
        out = {}
        for d in [enc_dict, z_dict, dec_dict]:
            out.update(d)
        return out


    def rec_loss(self, x_pred, x):
        """
        Reconstruction loss.
        """
        return F.mse_loss(x_pred, x, reduction="mean")


    def loss_function(
        self,
        x,
        x_pred,
        mu,
        logvar,
        **kwargs,
    ):
        """
        Calculates the CVAE loss (Reconstruction + KLD).
        """
        loss_dict = {}

        # 1. Reconstruction Loss
        MSE = self.rec_loss(x_pred, x)
        
        # 2. KLD Loss (from BaseCBVAE)
        KLD = self.KL_loss(mu, logvar)

        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        # 3. Total Loss
        loss_dict["Total_loss"] = MSE + self.beta * KLD

        # 4. Handle any extra losses from parent classes
        loss_dict = self._extra_loss(
            loss_dict,
            x=x,
            x_pred=x_pred,
            mu=mu,
            logvar=logvar,
            **kwargs,
        )

        return loss_dict

    
    def train_loop(self, data: torch.Tensor,
               concepts: torch.Tensor,
               num_epochs: int,
               batch_size: int,
               lr: float = 3e-4,
               lr_gamma: float = 0.997):
        """
        Defines the training loop for the scCVAE model.
        
        MODIFIED: Removed F1 score calculation, as this model
        does not predict concepts.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        torch.set_flush_denormal(True) # Add this to prevent slowdowns
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting CVAE training on {device} for {num_epochs} epochs...")

        self.train() # Set the model to training mode
        
        # --- 2. The Training Loop ---
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=150)

        for epoch in pbar:
            total_loss = 0.0
            total_rec_loss = 0.0
            total_kl_loss = 0.0
            
            for x_batch, concepts_batch in data_loader:
                # Move batch to the correct device
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)

                # --- Core logic ---
                # 1. Forward pass - concepts are now an INPUT
                out = self.forward(x_batch, concepts=concepts_batch)

                # 2. Calculate loss
                # Pass x_batch (as 'x') and the model output
                loss_dict = self.loss_function(x_batch, **out)
                loss = loss_dict["Total_loss"]
                # -------------------------------------------

                # 3. Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate losses for logging
                total_loss += loss.item()
                total_rec_loss += loss_dict.get("rec_loss", 0.0)
                total_kl_loss += loss_dict.get("KL_loss", 0.0)
                
                # --- F1 Score calculation is REMOVED ---

           # --- End of Epoch ---
            avg_loss = total_loss / len(data_loader)
            avg_rec = total_rec_loss / len(data_loader)
            avg_kl = total_kl_loss / len(data_loader)
             
            # --- Update progress bar (F1 score removed) ---
            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.3e}",
                "rec_loss": f"{avg_rec:.3e}",
                "kl_loss": f"{avg_kl:.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.5e}"
            })
             
            scheduler.step()
             
        print("Training finished.")
        self.eval() # Set the model to evaluation mode