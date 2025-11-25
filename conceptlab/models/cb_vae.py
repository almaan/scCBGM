import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
from scipy.sparse import csr_array, issparse

from tqdm import tqdm

from .base import BaseCBVAE
from .utils import sigmoid
from .encoder import EncoderBlock
from .decoder import DecoderBlock, SkipDecoderBlock, NoResDecoderBlock

EPS = 1e-6

import conceptlab as clab
import pytorch_lightning as pl
import numpy as np
import anndata as ad
from omegaconf import OmegaConf


class CB_VAE(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = EncoderBlock,
        _decoder: nn.Module = SkipDecoderBlock,
        **kwargs,
    ):

        super().__init__(
            config,
            **kwargs,
        )

        self.dropout = config.get("dropout", 0.0)
        self.variational = config.get("variational", True)

        # Encoder
        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            variational=self.variational,
        )

        self.use_cosine_loss = config.get("use_cosine_loss", False)

        if self.use_cosine_loss:
            print("Using Cosine Loss: Setting n_unknown to n_concepts")
            n_unknown = self.n_concepts
        elif "n_unknown" in config:
            n_unknown = config["n_unknown"]
        elif "min_bottleneck_size" in config:
            n_unknown = max(config.min_bottleneck_size, self.n_concepts)
        else:
            n_unknown = 32

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

        cb_unk_layers.append(nn.Linear(self.latent_dim, n_unknown))
        cb_unk_layers.append(nn.ReLU())

        self.cb_concepts_layers = nn.Sequential(*cb_concepts_layers)
        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_layers=self.n_layers,
            n_concepts=self.n_concepts,
            n_unknown=n_unknown,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

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

        if self.use_cosine_loss:
            self.orthogonality_loss = self._cosine_loss
        else:
            self.orthogonality_loss = self._cov_loss

    @property
    def has_concepts(
        self,
    ):
        return True

    def _extra_loss(self, loss_dict, *args, **kwargs):
        return loss_dict

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)

    def decode(self, h, **kwargs):

        return self._decoder(h, **kwargs)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):
        known_concepts = self.cb_concepts_layers(z)
        unknown = self.cb_unk_layers(z)

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts == None:
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

        device = self._encoder.x_embedder.weight.device

        enc = self.encode(x.to(device))
        enc["logvar"] = None
        z = self.reparametrize(**enc)
        cbm = self.cbm(
            **z,
            **enc,
            concepts=concepts.to(device),
            mask=mask.to(device),
            intervene=True,
        )

        dec = self.decode(**cbm)
        return dec

    def _cosine_loss(self, c, u):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(c, u))
        return output.mean()

    def _cov_loss(self, c, u):

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

    def loss_function(
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        # concept_proj,
        unknown,
        **kwargs,
    ):
        loss_dict = {}

        MSE = self.rec_loss(x_pred, x)

        if self.variational:
            KLD = self.KL_loss(mu, logvar)
        else:
            KLD = 0.0

        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        loss_dict["Total_loss"] = MSE + self.beta * KLD

        pred_concept_clipped = t.clip(pred_concept, 0, 1)

        if self.use_concept_loss:
            overall_concept_loss = self.concept_loss(pred_concept_clipped, concepts)
            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:
            orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
            loss_dict["orth_loss"] = orth_loss
            loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss

        loss_dict = self._extra_loss(
            loss_dict,
            x=x,
            concepts=concepts,
            x_pred=x_pred,
            mu=mu,
            logvar=logvar,
            pred_concept=pred_concept,
            unknown=unknown,
            **kwargs,
        )

        return loss_dict

    def train_loop(
        self,
        data: torch.Tensor,
        concepts: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        lr_gamma: float = 0.997,
        num_workers: int = 0,
    ):
        """
        Defines the training loop for the scCBGM model.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        torch.set_flush_denormal(True)  # Add this to prevent slowdowns

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        self.train()  # Set the model to training mode

        # --- 2. The Training Loop ---
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=150)

        for epoch in pbar:
            total_loss = 0.0

            # --- CHANGE: Initialize counters for F1 score calculation ---
            epoch_tp = 0.0
            epoch_fp = 0.0
            epoch_fn = 0.0
            # --- End of Change ---

            for x_batch, concepts_batch in data_loader:
                # Move batch to the correct device
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)

                # --- Core logic from your _step function ---
                # 1. Forward pass
                # We assume independent_training=True for this standalone loop
                out = self.forward(x_batch)

                # 2. Calculate loss
                loss_dict = self.loss_function(x_batch, concepts_batch, **out)
                loss = loss_dict["Total_loss"]
                # -------------------------------------------

                # 3. Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate losses for logging
                total_loss += loss.item()

                # --- CHANGE: Calculate and accumulate TP, FP, FN for the batch ---
                with torch.no_grad():
                    pred_concepts = out["pred_concept"]
                    predicted_labels = (pred_concepts > 0.5).float()
                    true_labels = concepts_batch

                    # True Positives: predicted is 1 and true is 1
                    epoch_tp += (predicted_labels * true_labels).sum().item()
                    # False Positives: predicted is 1 and true is 0
                    epoch_fp += (predicted_labels * (1 - true_labels)).sum().item()
                    # False Negatives: predicted is 0 and true is 1
                    epoch_fn += ((1 - predicted_labels) * true_labels).sum().item()
                # --- End of Change ---

            # --- End of Epoch ---
            avg_loss = total_loss / len(data_loader)

            # --- CHANGE: Calculate epoch-level F1 score and update progress bar ---
            epsilon = 1e-7  # To avoid division by zero
            precision = epoch_tp / (epoch_tp + epoch_fp + epsilon)
            recall = epoch_tp / (epoch_tp + epoch_fn + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

            pbar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.3e}",
                    "concept_f1": f"{f1_score:.4f}",  # Display F1 score
                    "lr": f"{scheduler.get_last_lr()[0]:.5e}",
                }
            )
            # --- End of Change ---

            scheduler.step()

        print("Training finished.")
        self.eval()  # Set the model to evaluation mode


class scCBGM(CB_VAE):
    def __init__(self, config, **kwargs):

        decoder_type = config.get("decoder_type", "skip")

        if decoder_type == "skip":
            print("Using Skip Decoder")
            decoder = SkipDecoderBlock
        elif decoder_type == "no residual":
            print("Using No Residual Decoder")
            decoder = NoResDecoderBlock
        else:
            print("Using Residual Decoder")
            decoder = DecoderBlock
        super().__init__(config, _decoder=decoder, **kwargs)

    def decode(self, input_concept, unknown, **kwargs):
        return self._decoder(input_concept, unknown, **kwargs)


class CBM_MetaTrainer:
    def __init__(
        self,
        cbm_config,
        max_epochs,
        log_every_n_steps,
        concept_key,
        num_workers,
        obsm_key: str = "X",
        z_score: bool = False,
    ):
        """
        Class to train and predict interventions with a scCBMG model
        Inputs:
        - cbm_config: config for the model
        - max_epochs: max number of epochs to train for
        - log_every_n_steps:
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key where to apply the method on (obsm) - "X" or "X_pca"
        - zscore: whether to whiten the data
        """
        self.cbm_config = cbm_config
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.concept_key = concept_key
        self.num_workers = num_workers

        self.model = None

        self.obsm_key = obsm_key
        self.z_score = z_score

    def train(self, adata_train):
        """Trains and returns the scCBGM model."""
        print("Training scCBGM model...")

        if self.obsm_key != "X":
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            # if self.z_score:
            #    data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization

        torch.set_flush_denormal(True)

        config = OmegaConf.create(
            dict(
                input_dim=data_matrix.shape[1],
                n_concepts=adata_train.obsm[self.concept_key].shape[1],
            )
        )
        merged_config = OmegaConf.merge(config, self.cbm_config)

        print("using model config")
        print(merged_config)

        model = clab.models.scCBGM(merged_config)

        model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(
                adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)
            ),
            num_epochs=self.max_epochs,
            batch_size=128,
            num_workers=self.num_workers,
        )

        self.model = model

        return self.model

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):
        """Performs intervention using a trained scCBGM model.
        Returns an anndata with predicted values."""
        print("Performing intervention with scCBGM...")

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

        # Define the intervention by creating a mask and new concept values
        mask = torch.zeros(c_intervene_on.shape, dtype=torch.float32)
        mask[:, concept_to_intervene_idx] = 1

        inter_concepts = torch.tensor(c_intervene_on, dtype=torch.float32)
        inter_concepts[:, concept_to_intervene_idx] = (
            1 - inter_concepts[:, concept_to_intervene_idx]
        )  # Set stim concept to the opposite of the observed value.

        with torch.no_grad():
            inter_preds = self.model.intervene(
                x_intervene_on.to("cuda"),
                mask=mask.to("cuda"),
                concepts=inter_concepts.to("cuda"),
            )

        inter_preds = inter_preds["x_pred"].cpu().numpy()

        pred_adata = adata_inter.copy()

        if self.obsm_key != "X":
            x_inter_preds = np.zeros_like(adata_inter.X)
            pred_adata.obsm[self.obsm_key] = inter_preds
        else:
            x_inter_preds = inter_preds

        pred_adata.X = x_inter_preds
        pred_adata.obs["ident"] = "intervened on"
        pred_adata.obs["cell_stim"] = hold_out_label + "*"

        return pred_adata


class Mixed_CBM_MetaTrainer:
    def __init__(
        self,
        cbm_config,
        max_epochs,
        log_every_n_steps,
        concept_key,
        num_workers,
        obsm_key: str = "X",
        z_score: bool = False,
        hard_concept_key: str = None,
        soft_concept_key: str = None,
    ):
        """
        Class to train and predict interventions with a scCBMG model
        Inputs:
        - cbm_config: config for the model
        - max_epochs: max number of epochs to train for
        - log_every_n_steps:
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key where to apply the method on (obsm) - "X" or "X_pca"
        - zscore: whether to whiten the data
        """
        self.cbm_config = cbm_config
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.concept_key = concept_key
        self.num_workers = num_workers

        self.model = None

        self.obsm_key = obsm_key
        self.z_score = z_score

        self.hard_concept_key = hard_concept_key
        self.soft_concept_key = soft_concept_key
        if self.hard_concept_key is None and self.soft_concept_key is None:
            raise ValueError(
                "You must provide at least one of 'hard_concept_key' or 'soft_concept_key'."
            )

    def train(self, adata_train):
        """Trains and returns the scCBGM model."""
        print("Training scCBGM model...")

        if self.obsm_key != "X":
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            # if self.z_score:
            #    data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization

        torch.set_flush_denormal(True)

        # --- Prepare Concept Tensors ---
        hard_concepts_tensor = None
        n_hard = 0
        if self.hard_concept_key:
            hard_concepts_data = (
                adata_train.obsm[self.hard_concept_key].to_numpy().astype(np.float32)
            )
            hard_concepts_tensor = torch.from_numpy(hard_concepts_data)
            n_hard = hard_concepts_tensor.shape[1]

        soft_concepts_tensor = None
        n_soft = 0
        if self.soft_concept_key:
            soft_concepts_data = (
                adata_train.obsm[self.soft_concept_key].to_numpy().astype(np.float32)
            )
            soft_concepts_tensor = torch.from_numpy(soft_concepts_data)
            n_soft = soft_concepts_tensor.shape[1]

        config = OmegaConf.create(
            dict(
                input_dim=data_matrix.shape[1],
                n_concepts=adata_train.obsm[self.concept_key].shape[1],
                use_soft_concepts=self.soft_concept_key is not None,
                use_hard_concepts=self.hard_concept_key is not None,
                n_hard_concepts=n_hard,
                n_soft_concepts=n_soft,
            )
        )
        merged_config = OmegaConf.merge(config, self.cbm_config)

        model = clab.models.CB_VAE_MIXED(merged_config)

        print("\nModel's train_loop would be called with:")
        print(f"data shape: {data_matrix.shape}")
        if hard_concepts_tensor is not None:
            print(f"hard_concepts shape: {hard_concepts_tensor.shape}")
        if soft_concepts_tensor is not None:
            print(f"soft_concepts shape: {soft_concepts_tensor.shape}")

        model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            hard_concepts=hard_concepts_tensor,
            soft_concepts=soft_concepts_tensor,
            # concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
            num_epochs=self.max_epochs,
            batch_size=128,
            num_workers=self.num_workers,
        )

        self.model = model

        return self.model

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):
        """Performs intervention using a trained scCBGM model.
        Returns an anndata with predicted values."""
        print("Performing intervention with scCBGM...")

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

        # --- Prepare original concept tensors and find column indices ---
        concept_parts, hard_concepts_df, soft_concepts_df = [], None, None
        n_hard = 0

        if self.hard_concept_key:
            hard_concepts_df = adata_inter.obsm[self.hard_concept_key]
            concept_parts.append(
                torch.from_numpy(hard_concepts_df.to_numpy(dtype=np.float32))
            )
            n_hard = hard_concepts_df.shape[1]

        if self.soft_concept_key:
            soft_concepts_df = adata_inter.obsm[self.soft_concept_key]
            concept_parts.append(
                torch.from_numpy(soft_concepts_df.to_numpy(dtype=np.float32))
            )

        if not concept_parts:
            raise ValueError(
                "Must provide at least one concept key ('hard_concept_key' or 'soft_concept_key') to perform intervention."
            )

        c_intervene_on = torch.cat(concept_parts, dim=1)
        inter_concepts = c_intervene_on.clone()
        mask = torch.zeros_like(c_intervene_on)

        # --- Build the mask and intervention tensor from the dictionary ---
        for concept_name, new_value in zip(concepts_to_flip, values_to_set):
            found = False
            if (
                hard_concepts_df is not None
                and concept_name in hard_concepts_df.columns
            ):
                col_idx = hard_concepts_df.columns.get_loc(concept_name)
                mask[:, col_idx] = 1
                inter_concepts[:, col_idx] = new_value
                found = True
                print(
                    f"Intervening on HARD concept '{concept_name}' (index {col_idx}) -> {new_value}"
                )

            elif (
                soft_concepts_df is not None
                and concept_name in soft_concepts_df.columns
            ):
                col_idx = soft_concepts_df.columns.get_loc(concept_name) + n_hard
                mask[:, col_idx] = 1
                inter_concepts[:, col_idx] = new_value
                found = True
                print(
                    f"Intervening on SOFT concept '{concept_name}' (index {col_idx}) -> {new_value}"
                )

            if not found:
                raise ValueError(
                    f"Warning: Concept '{concept_name}' not found in provided keys. Ignoring."
                )

        # --- Run intervention on the model ---
        device = "cuda"
        with torch.no_grad():
            inter_preds_dict = self.model.intervene(
                x_intervene_on.to(device),
                mask=mask.to(device),
                concepts=inter_concepts.to(device),
            )
        inter_preds = inter_preds_dict["x_pred"].cpu().numpy()

        # --- Create prediction AnnData object ---
        pred_adata = adata_inter.copy()

        pred_adata.obs["ident"] = "intervened"

        if self.obsm_key != "X":
            pred_adata.X = np.zeros_like(pred_adata.X)
            pred_adata.obsm[self.obsm_key] = inter_preds
        else:
            pred_adata.X = inter_preds

        return pred_adata
