import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb
from tqdm import tqdm
import numpy as np

from .base import BaseCBVAE
from .utils import sigmoid
from .encoder import DefaultEncoderBlock
from .decoder import DefaultDecoderBlock, SkipDecoderBlock

from torch.utils.data import TensorDataset, DataLoader

from omegaconf import OmegaConf
import conceptlab as clab
import anndata as ad

class CEM_VAE(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = DefaultEncoderBlock,
        _decoder: nn.Module = DefaultDecoderBlock,
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
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        if "n_unknown" in config:
            n_unknown = config["n_unknown"]
        elif "min_bottleneck_size" in config:
            n_unknown = max(config.min_bottleneck_size, self.n_concepts)
        else:
            n_unknown = 32

        if self.n_concepts > n_unknown:
            # Case 1: If n_concepts is larger than n_unknown
            n_unknown = self.n_concepts
            self.emb_size = 1
        else:
            # Case 2: Make n_concept * emb_size = n_unknown
            # Since we can't modify n_concepts, we'll adjust emb_size
            self.emb_size = n_unknown // self.n_concepts
            # Adjust n_unknown to be exactly divisible by n_concepts
            n_unknown = self.n_concepts * self.emb_size

        # Create separate context generators for each concept
        self.concept_context_generators = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.latent_dim, 2 * self.emb_size), nn.ReLU())
                for _ in range(self.n_concepts)
            ]
        )

        # Separate probability generator for each concept
        self.concept_prob_generators = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * self.emb_size, 1), nn.Sigmoid())
                for _ in range(self.n_concepts)
            ]
        )

        if "cb_layers" in config:
            cb_layers = config["cb_layers"]
        else:
            cb_layers = 1

        cb_unk_layers = []

        for k in range(0, cb_layers - 1):

            layer_k = [
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]

            cb_unk_layers += layer_k

        cb_unk_layers.append(nn.Linear(self.latent_dim, n_unknown))
        cb_unk_layers.append(nn.ReLU())

        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_concepts=n_unknown,
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

        self.use_concept_loss = config.get("use_concept_loss", True)

        #self.save_hyperparameters()

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

    def concept_loss(self, pred_concept, concepts, **kwargs):

        overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        unknown = self.cb_unk_layers(z)

        # Lists to store results
        contexts = []
        probs = []

        # Generate context and probability for each concept
        for i in range(self.n_concepts):
            # Generate context
            context = self.concept_context_generators[i](
                z
            )  # shape: [..., 2 * emb_size]
            contexts.append(context)

            # Generate probability from this concept's context
            prob = self.concept_prob_generators[i](context)  # shape: [..., 1]
            probs.append(prob)

        # Stack results
        contexts = torch.stack(
            contexts, dim=-2
        )  # shape: [..., n_concepts, 2 * emb_size]
        contexts = contexts.unsqueeze(-2) if contexts.ndimension() == 2 else contexts

        known_concepts = torch.stack(probs, dim=-2).squeeze(
            -1
        )  # Ensure shape [..., n_concepts]
        known_concepts = (
            known_concepts.unsqueeze(-1)
            if known_concepts.ndimension() == 1
            else known_concepts
        )

        pos_context = context[..., : self.emb_size]  # shape: [..., emb_size]
        neg_context = context[..., self.emb_size :]  # shape: [..., emb_size]

        # Expand contexts to match number of concepts
        pos_context = pos_context.unsqueeze(-2).expand(
            *pos_context.shape[:-1], self.n_concepts, self.emb_size
        )
        neg_context = neg_context.unsqueeze(-2).expand(
            *neg_context.shape[:-1], self.n_concepts, self.emb_size
        )

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts == None:
                input_concept = known_concepts
            else:
                input_concept = concepts

        input_concept = input_concept.unsqueeze(-1).expand(
            *input_concept.shape, self.emb_size
        )
        # Weight contexts with probabilities
        weighted_pos = pos_context * input_concept
        weighted_neg = neg_context * (1 - input_concept)

        # Combine weighted contexts
        combined = weighted_pos + weighted_neg  # shape: [..., n_concepts, emb_size]
        # Reshape to have size emb_size * n_concepts

        final_shape = list(combined.shape[:-2]) + [self.emb_size * self.n_concepts]

        emd_concept = combined.reshape(*final_shape)
        h = torch.cat((emd_concept, unknown), 1)

        return dict(
            pred_concept=known_concepts,
            emd_concept=emd_concept,
            unknown=unknown,
            h=h,
        )

    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm)
        return dec

    def orthogonality_loss(self, c, u):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(c, u))
        return output.mean()

    def rec_loss(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction="mean")

    def loss_function(
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        emd_concept,
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
            orth_loss = self.orthogonality_loss(emd_concept, unknown)
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

    def configure_optimizers(
        self,
    ):

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
        # Define the CosineAnnealingLR scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.997)

        # Return a dictionary with the optimizer and the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # The LR scheduler instance
                "interval": "epoch",  # The interval to step the scheduler ('epoch' or 'step')
                "frequency": 1,  # How often to update the scheduler
                "monitor": "val_loss",  # Optional: if you use reduce-on-plateau schedulers
            },
        }
    def train_loop(self, data: torch.Tensor,
               concepts: torch.Tensor,
               num_epochs: int,
               batch_size: int,
               lr_gamma: float = 0.997,
               num_workers:int = 0):
        """
        Defines the training loop for the CEM-VAE model.
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
            epsilon = 1e-7 # To avoid division by zero
            precision = epoch_tp / (epoch_tp + epoch_fp + epsilon)
            recall = epoch_tp / (epoch_tp + epoch_fn + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.3e}",
                "concept_f1": f"{f1_score:.4f}", # Display F1 score
                "lr": f"{scheduler.get_last_lr()[0]:.5e}"
            })
            # --- End of Change ---
             
            scheduler.step()
             

        print("Training finished.")
        self.eval() # Set the model to evaluation mode
    

class CEM_MetaTrainer:

    def __init__(self,
                 cbm_config,
                 max_epochs,
                 log_every_n_steps,
                concept_key,
                num_workers,
                batch_size,
                obsm_key:str = "X",
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
        - obsm_key where to apply the method on (obsm) - "X" or "X_pca"
        - zscore: whether to whiten the data
        """
        self.cbm_config = cbm_config
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.concept_key = concept_key
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.model = None

        self.obsm_key = obsm_key
        self.z_score = z_score
        
    def train_loop(self, data_module, model):    
        trainer = pl.Trainer(
            max_epochs=self.max_epochs
            )

        trainer.fit(model, data_module)

        model.eval()
        return model
    
    def train(self, adata_train):

        """Trains and returns the scCBGM model."""
        print("Training scCBGM model...")

        if self.obsm_key != "X":
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            #if self.z_score:
            #    data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization       
 
        torch.set_flush_denormal(True)

        config = OmegaConf.create(dict(
            input_dim=data_matrix.shape[1], 
            n_concepts=adata_train.obsm[self.concept_key].shape[1],
        ))
        merged_config = OmegaConf.merge(config, self.cbm_config)
        
        model = clab.models.CEM_VAE(merged_config)

        model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
            num_epochs=self.max_epochs, batch_size=self.batch_size,
            num_workers=self.num_workers
            )
        
        self.model = model

        return self.model

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip):
        """Performs intervention using a trained CEM-VAE model.
        Returns an anndata with predicted values."""
        print("Performing intervention with CEM-VAE...")

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() before predict_intervention().")
        
        if self.obsm_key != "X":
            x_intervene_on =  torch.tensor(adata_inter.obsm[self.obsm_key], dtype=torch.float32)
        else:
            x_intervene_on = torch.tensor(adata_inter.X, dtype=torch.float32)

        c_intervene_on = adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32)

        # what indices should we flip in the concepts
        concept_to_intervene_idx = [idx for idx,c in enumerate(adata_inter.obsm[self.concept_key].columns) if c in concepts_to_flip]

        # Define the intervention by creating a mask and new concept values
        mask = torch.zeros(c_intervene_on.shape, dtype=torch.float32)
        mask[:, concept_to_intervene_idx] = 1 

        inter_concepts = torch.tensor(c_intervene_on, dtype=torch.float32)
        inter_concepts[:, concept_to_intervene_idx] = 1 - inter_concepts[:, concept_to_intervene_idx] # Set stim concept to the opposite of the observed value.

        with torch.no_grad():
            inter_preds = self.model.intervene(x_intervene_on.to("cuda"), mask=mask.to("cuda"), concepts=inter_concepts.to("cuda"))
        
        inter_preds = inter_preds['x_pred'].cpu().numpy() 

        if(self.obsm_key != "X"):
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds

        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs['ident'] = 'intervened on'
        pred_adata.obs['cell_stim'] = hold_out_label + '*'

        if(self.obsm_key != "X"):
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata