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

import numpy as np
import anndata as ad
from omegaconf import OmegaConf
import conceptlab as clab
import hydra


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
        self.n_concepts = config.n_concepts  # Get n_concepts from config
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
            n_concepts=self.n_concepts,
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
            dropout=self.dropout,
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
            raise ValueError(
                "CVAE requires 'concepts' (the condition) to be provided during forward pass."
            )

        # 1. Prepare encoder input
        # MODIFIED: No longer concatenating here.
        # enc_input = torch.cat((x, concepts), 1)

        # 2. Encode
        # MODIFIED: Pass 'x' and 'concepts' as separate arguments
        enc_dict = self.encode(x, concepts)

        # 3. Reparameterize
        z_dict = self.reparametrize(**enc_dict)  # From BaseCBVAE

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
            raise ValueError(
                "CVAE intervention requires both 'concepts_enc' and 'concepts_dec' to be provided."
            )

        # 1. Encode x conditioned on 'concepts_enc'
        enc_dict = self.encode(x, concepts=concepts_enc, **kwargs)

        # 2. Reparameterize to get z
        z_dict = self.reparametrize(**enc_dict)  # From BaseCBVAE

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

    def train_loop(
        self,
        data: torch.Tensor,
        concepts: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        lr: float = 3e-4,
        lr_gamma: float = 0.997,
        num_workers: int = 0,
    ):
        """
        Defines the training loop for the scCVAE model.

        MODIFIED: Removed F1 score calculation, as this model
        does not predict concepts.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        torch.set_flush_denormal(True)  # Add this to prevent slowdowns

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting CVAE training on {device} for {num_epochs} epochs...")

        self.train()  # Set the model to training mode

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
            pbar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.3e}",
                    "rec_loss": f"{avg_rec:.3e}",
                    "kl_loss": f"{avg_kl:.3e}",
                    "lr": f"{scheduler.get_last_lr()[0]:.5e}",
                }
            )

            scheduler.step()

        print("Training finished.")
        self.eval()  # Set the model to evaluation mode


class CVAE_MetaTrainer:
    def __init__(
        self,
        cvae_config,
        max_epochs,
        log_every_n_steps,
        concept_key,
        num_workers,
        obsm_key: str = "X",
        z_score: bool = False,
    ):
        """
        Class to train and predict interventions with a scCVAE model
        Inputs:
        - cvae_config: config for the model
        - max_epochs: max number of epochs to train for
        - log_every_n_steps:
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key where to apply the method on (obsm) - "X" or "X_pca"
        - zscore: whether to whiten the data
        """
        self.cvae_config = cvae_config
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.concept_key = concept_key
        self.num_workers = num_workers

        self.model = None

        self.obsm_key = obsm_key
        self.z_score = z_score

    def train(self, adata_train):
        """Trains and returns the scCVAE model."""
        print("Training scCVAE model...")

        if self.obsm_key != "X":
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X

        torch.set_flush_denormal(True)

        config = OmegaConf.create(
            dict(
                input_dim=data_matrix.shape[-1],
                n_concepts=adata_train.obsm[self.concept_key].shape[1],
            )
        )
        merged_config = OmegaConf.merge(config, self.cvae_config)

        print("using model config")
        print(merged_config)

        model = clab.models.sc_cvae.scCVAE(merged_config)

        model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(
                adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)
            ),
            num_epochs=self.max_epochs,
            batch_size=128,
            num_workers=self.num_workers,
            lr=self.cvae_config["lr"],
        )

        self.model = model

        return self.model

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):
        """Performs intervention using a trained scCVAE model.
        Returns an anndata with predicted values."""
        print("Performing intervention with scCVAE...")

        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call train() before predict_intervention()."
            )

        if self.obsm_key != "X":
            x_intervene_on = torch.tensor(
                adata_inter.obsm[self.obsm_key], dtype=torch.float32
            )
        else:
            x_intervene_on = torch.tensor(adata_inter.X, dtype=torch.float32)

        c_intervene_on = (
            adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32)
        )

        # what indices should we flip in the concepts
        concept_to_intervene_idx = [
            idx
            for idx, c in enumerate(adata_inter.obsm[self.concept_key].columns)
            if c in concepts_to_flip
        ]

        # Define the intervention by creating new concept values for encoding and decoding
        concepts_enc = torch.tensor(c_intervene_on, dtype=torch.float32)
        concepts_dec = torch.tensor(c_intervene_on, dtype=torch.float32)

        # Set the intervention concepts to the specified values
        if values_to_set is not None:
            for idx, value in zip(concept_to_intervene_idx, values_to_set):
                concepts_dec[:, idx] = value
        else:
            # Default: flip the concepts (0 -> 1, 1 -> 0)
            concepts_dec[:, concept_to_intervene_idx] = (
                1 - concepts_dec[:, concept_to_intervene_idx]
            )

        with torch.no_grad():
            inter_preds = self.model.intervene(
                x_intervene_on.to("cuda"),
                concepts_enc=concepts_enc.to("cuda"),
                concepts_dec=concepts_dec.to("cuda"),
            )

        inter_preds = inter_preds["x_pred"].cpu().numpy()

        if self.obsm_key != "X":
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds

        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs["ident"] = "intervened on"
        pred_adata.obs["cell_stim"] = hold_out_label + "*"

        if self.obsm_key != "X":
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata


class CVAEFM_MetaTrainer:
    def __init__(
        self,
        fm_mod_cfg,
        cvae_mod=None,
        num_epochs=1000,
        batch_size=128,
        lr=3e-4,
        concept_key="concepts",
        num_workers=0,
        obsm_key: str = "X",
        z_score: bool = False,
        edit: bool = True,
    ):
        """
        Class to train and predict interventions with a scCVAE-FM model
        Inputs:
        - cvae_mod: the scCVAE model underlying the model
        - fm_mod_cfg: the Flow Matching model configuration
        - num_epochs: max number of epochs to train for
        - batch_size: batch size for training
        - lr: learning rate
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key: key of the anndata obsm to train on.
        - z_score: whether to whiten the data
        - edit: whether to use encode/decode (True) or just decode (False) for intervention
        """
        self.scCVAE_model = cvae_mod
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.concept_key = concept_key
        self.num_workers = num_workers
        self.obsm_key = obsm_key
        self.z_score = z_score
        self.fm_mod_cfg = fm_mod_cfg
        self.edit = edit

    def get_cvae_latents(self, adata_full):
        """
        Uses a trained scCVAE to generate its latent representation ('mu')
        for all data, conditioned on the true concepts.

        It saves the latent 'mu' and a concatenation of 'mu' and the
        true concepts into adata_full.obsm.
        """
        print("Generating latents from scCVAE...")

        if self.scCVAE_model.model is None:
            raise ValueError("scCVAE model is not trained. Call fit_cvae_model first")

        self.scCVAE_model.model.to("cuda")  # Ensure model is on the correct device

        # --- 1. Get Input Data 'x' ---
        if self.obsm_key != "X":
            all_x = torch.tensor(
                adata_full.obsm[self.obsm_key], dtype=torch.float32
            ).to("cuda")
        else:
            all_x = torch.tensor(adata_full.X, dtype=torch.float32).to("cuda")

        # --- 2. Get Conditional Data 'concepts' ---
        # The CVAE encoder requires the concepts as an input condition
        all_concepts_numpy = (
            adata_full.obsm[self.concept_key].to_numpy().astype(np.float32)
        )
        all_concepts = torch.tensor(all_concepts_numpy, dtype=torch.float32).to("cuda")

        # --- 3. Get Latent Variable 'mu' ---
        with torch.no_grad():
            # Encode x *conditioned on* the true concepts
            enc_dict = self.scCVAE_model.model.encode(all_x, all_concepts)

            # This is the CVAE latent representation
            latent_mu = enc_dict["mu"].cpu().numpy()

        # --- 4. Save Outputs to AnnData ---
        adata_full.obsm["scCVAE_latent"] = latent_mu

        # Per request: save the latent concatenated with the true concepts
        adata_full.obsm["scCVAE_latent_and_concepts"] = np.concatenate(
            [latent_mu, all_concepts_numpy], axis=1
        )

        print("Saved 'scCVAE_latent' and 'scCVAE_latent_and_concepts' to adata.obsm")
        return adata_full

    def fit_cvae_model(self, adata):
        """Trains the scCVAE model."""
        self.scCVAE_model.train(adata)

    def train(self, adata_train):
        """Trains and returns the CVAE-FM model using CVAE latents."""
        print("Training CVAE-FM model with CVAE latents...")

        # First, fit the CVAE model
        self.fit_cvae_model(adata_train)

        # Generate CVAE latents
        adata_train = self.get_cvae_latents(adata_train.copy())

        # Get the data matrix
        if self.obsm_key != "X":
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X

        # Use the concatenated latent + concepts as conditioning
        latent_and_concepts_key = "scCVAE_latent_and_concepts"

        mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
        mod_cfg["input_dim"] = data_matrix.shape[1]
        mod_cfg["n_concepts"] = adata_train.obsm[latent_and_concepts_key].shape[1]
        self.fm_model = clab.models.cond_fm.Cond_FM(config=mod_cfg)

        self.fm_model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(
                adata_train.obsm[latent_and_concepts_key].astype(np.float32)
            ),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            num_workers=self.num_workers,
        )

        return self.fm_model

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):
        """Performs intervention using a trained CVAE-FM model.

        Args:
            adata_inter: AnnData object with intervention data
            hold_out_label: Label for held-out condition
            concepts_to_flip: List of concept names to flip
            values_to_set: Optional list of values to set for each concept
        """
        print("Performing intervention with CVAE-FM...")

        # Generate CVAE latents for intervention data
        adata_inter = self.get_cvae_latents(adata_inter.copy())

        # Find which concept indices to intervene on
        concept_to_intervene_idx = [
            idx
            for idx, c in enumerate(adata_inter.obsm[self.concept_key].columns)
            if c in concepts_to_flip
        ]

        # Get the data matrix
        if self.obsm_key != "X":
            x_inter = adata_inter.obsm[self.obsm_key]
        else:
            x_inter = adata_inter.X

        # Get the latent + concepts representation
        latent_and_concepts_key = "scCVAE_latent_and_concepts"
        init_concepts = adata_inter.obsm[latent_and_concepts_key].astype(np.float32)
        edit_concepts = init_concepts.copy()

        # The concepts are at the end of the concatenated vector
        # We need to offset by the latent dimension
        latent_dim = adata_inter.obsm["scCVAE_latent"].shape[1]

        # Apply interventions to the concept part (after latent)
        if values_to_set is not None:
            for idx, value in zip(concept_to_intervene_idx, values_to_set):
                edit_concepts[:, latent_dim + idx] = value
        else:
            # Default: flip the concepts (0 -> 1, 1 -> 0)
            for idx in concept_to_intervene_idx:
                edit_concepts[:, latent_dim + idx] = (
                    1 - edit_concepts[:, latent_dim + idx]
                )

        # Perform intervention using Flow Matching
        if self.edit:
            inter_preds = self.fm_model.edit(
                x=torch.from_numpy(x_inter.astype(np.float32)).to("cuda"),
                c=torch.from_numpy(init_concepts).to("cuda"),
                c_prime=torch.from_numpy(edit_concepts).to("cuda"),
                t_edit=0.0,
                n_steps=1000,
                w_cfg_forward=1.0,
                w_cfg_backward=1.0,
                noise_add=0.0,
            )
        else:
            inter_preds = self.fm_model.decode(
                h=torch.from_numpy(edit_concepts).to("cuda"),
                n_steps=1000,
                w_cfg=1.0,
            )

        inter_preds = inter_preds.detach().cpu().numpy()

        # Create output AnnData
        if self.obsm_key != "X":
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds

        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs["ident"] = "intervened on"
        pred_adata.obs["cell_stim"] = hold_out_label + "*"

        if self.obsm_key != "X":
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata
