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
from .decoder import DecoderBlock

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
        _decoder: nn.Module = DecoderBlock,
        **kwargs,
    ):

        super().__init__(
            config,
            **kwargs,
        )

        self.dropout = config.get("dropout", 0.0)

        # Encoder

        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        if "n_unknown" in config:
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
            n_layers = self.n_layers,
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
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts.to(device), mask=mask.to(device), intervene=True)
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
        KLD = self.KL_loss(mu, logvar)

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

    
    def train_loop(self, data: torch.Tensor,
               concepts: torch.Tensor,
               num_epochs: int,
               batch_size: int,
               lr_gamma: float = 0.997,
               num_workers:int = 0):
        """
        Defines the training loop for the scCBGM model.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        torch.set_flush_denormal(True) # Add this to prevent slowdowns
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

        lr = self.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        self.train() # Set the model to training mode
        
        # --- 2. The Training Loop ---
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)
        history = {"total_loss": [], "rec_loss": [], "kl_loss": [], "concept_loss": [], "orth_loss": []}

        for epoch in pbar:
            epoch_losses = {key: 0.0 for key in history.keys()}

            for x_batch, concepts_batch in data_loader:
                # Move batch to the correct device
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)

                # --- Core logic from your _step function ---
                # 1. Forward pass
                # We assume independent_training=True for this standalone loop
                out = self.forward(x_batch, concepts_batch)

                # 2. Calculate loss
                loss_dict = self.loss_function(x_batch, concepts_batch, **out)
                total_loss = loss_dict["Total_loss"]
                # -------------------------------------------

                # 3. Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate losses for logging
                for key in epoch_losses.keys():
                     epoch_losses[key] += loss_dict.get(key.replace("_loss", "") + "_loss", torch.tensor(0.0)).item()

            # --- 3. End of Epoch ---
            # Update learning rate
            scheduler.step()

            # Log average losses for the epoch
            num_batches = len(data_loader)
            for key in history.keys():
                history[key].append(epoch_losses[key] / num_batches)
            
            pbar.set_postfix({
                "Loss": history['total_loss'][-1],
                "LR": scheduler.get_last_lr()[0]
            })

        print("Training finished.")
        self.eval() # Set the model to evaluation mode
        return history

class scCBGM(CB_VAE):
    def __init__(self, config, **kwargs):
        super().__init__(config, _decoder=DecoderBlock, **kwargs)

    def decode(self, input_concept, unknown, **kwargs):
        return self._decoder(input_concept, unknown, **kwargs)


class CBM_MetaTrainer:

    def __init__(self,
                 cbm_config,
                 max_epochs,
                 log_every_n_steps,
                concept_key,
                num_workers,
                pca:bool = False,
                z_score:bool = False
            ):
        """
        Class to train and predict interventions with a scCBMG model
        Inputs:
        - cbm_config: config for the model
        - max_epochs: max number of epochs to train for
        - log_every_n_steps: 
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - pca: whether to train, and predict in PCA space.
        - zscore: whether to whiten the data
        """
        self.cbm_config = cbm_config
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.concept_key = concept_key
        self.num_workers = num_workers

        self.model = None

        self.pca = pca
        self.z_score = z_score
    
    def train(self, adata_train):

        """Trains and returns the scCBGM model."""
        print("Training scCBGM model...")

        if self.pca:
            data_matrix = adata_train.obsm['X_pca']
        else:
            data_matrix = adata_train.X
            if self.z_score:
                data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization       

        if issparse(data_matrix):
            data_matrix = data_matrix.toarray()
        
        torch.set_flush_denormal(True)

        config = OmegaConf.create(dict(
            input_dim=adata_train.shape[1], 
            n_concepts=adata_train.obsm[self.concept_key].shape[1],
        ))
        merged_config = OmegaConf.merge(config, self.cbm_config)
        
        model = clab.models.scCBGM(merged_config)

        model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
            num_epochs=self.max_epochs, batch_size=128,
            num_workers = self.num_workers
            )
        
        self.model = model

        #data_module = clab.data.dataloader.GeneExpressionDataModule(
        #    adata_train, add_concepts=True, concept_key=self.concept_key, batch_size=512, normalize=False, num_workers=self.num_workers
        #)

        #trainer = pl.Trainer(max_epochs=self.max_epochs, log_every_n_steps = self.log_every_n_steps, accelerator='auto')
        #trainer.fit(model, data_module)

        #self.model = model.to("cpu").eval()

        return self.model

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip):
        """Performs intervention using a trained scCBGM model.
        Returns an anndata with predicted values."""
        print("Performing intervention with scCBGM...")

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() before predict_intervention().")
        
        if self.pca:
            x_intervene_on =  torch.tensor(adata_inter.obsm['X_pca'], dtype=torch.float32)
        else:
            if isinstance(adata_inter.X, np.ndarray):
                x_intervene_on = torch.tensor(adata_inter.X, dtype=torch.float32)
            else:
                x_intervene_on = torch.tensor(adata_inter.X.toarray(), dtype=torch.float32)
        c_intervene_on = adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32)

        # what indices should we flip in the concepts
        concept_to_intervene_idx = [idx for idx,c in enumerate(adata_inter.obsm[self.concept_key].columns) if c in concepts_to_flip]

        # Define the intervention by creating a mask and new concept values
        mask = torch.zeros(c_intervene_on.shape, dtype=torch.float32)
        mask[:, concept_to_intervene_idx] = 1  # Intervene on the last concept (stim)

        inter_concepts = torch.tensor(c_intervene_on, dtype=torch.float32)
        inter_concepts[:, concept_to_intervene_idx] = 1 - inter_concepts[:, concept_to_intervene_idx] # Set stim concept to the opposite of the observed value.

        with torch.no_grad():
            inter_preds = self.model.intervene(x_intervene_on, mask=mask, concepts=inter_concepts)
        
        inter_preds = inter_preds['x_pred'].cpu().numpy() 

        if(self.pca):
            x_inter_preds = adata_inter.uns['pc_transform'].inverse_transform(inter_preds)
        else:
            x_inter_preds = inter_preds

        pred_adata = ad.AnnData(x_inter_preds, var=adata_inter.var)
        pred_adata.obs['ident'] = 'intervened on'
        pred_adata.obs['cell_stim'] = hold_out_label + '*'

        if(self.pca):
            pred_adata.obsm['X_pca'] = inter_preds

        return pred_adata