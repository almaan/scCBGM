import torch
import torch.nn as nn
import torch.optim.lr_scheduler 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import conceptlab as clab
import pytorch_lightning as pl
import numpy as np
import anndata as ad
import hydra
import pandas as pd

# Assuming the FlowDecoder model is defined in FlowDecoder.py as specified previously
from .mlp import FlowDecoder
from .utils import optimal_transport_coupling


class CB_FM:
    """
    Implements a conditional flow matching model using a base FlowDecoder.
    This class handles training and sampling with optional "known" and "unknown" concepts.
    """

    def __init__(self, x_dim: int, c_known_dim: int = 0, c_unknown_dim: int = 0, emb_dim: int = 256, n_layers: int = 4, dropout: float = 0.1, num_workers = 0):
        """
        Initializes the FlowMatchingModel.
        Args:
            x_dim (int): The dimensionality of the input data (e.g., cell features).
            c_known_dim (int): The dimensionality of the "known" conditioning vector.
            c_unknown_dim (int): The dimensionality of the "unknown" conditioning vector.
            emb_dim (int): The embedding dimension for the FlowDecoder.
            n_layers (int): The number of layers in the FlowDecoder.
            dropout (float): The dropout rate for the FlowDecoder.
        """
        self.x_dim = x_dim
        self.c_known_dim = c_known_dim
        self.c_unknown_dim = c_unknown_dim
        self.c_dim = c_known_dim + c_unknown_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = num_workers
        
        if self.c_dim == 0:
            print("Warning: All concept dimensions are 0. Model will be purely unconditional.")

        self.model = FlowDecoder(
            x_dim=x_dim,
            c_dim=self.c_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)

    def _get_train_step_data(self, x1: torch.Tensor, ot: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Generates tensors for a single training step based on the flow matching formula.
        """
        x0 = torch.randn_like(x1)
        cost_improvement = 1.0
        if ot:
            x0, cost_improvement = optimal_transport_coupling(x0, x1)
        
        t = torch.rand(x1.shape[0], device=self.device)
        t_reshaped = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1
        ut = x1 - x0
        
        return t, xt, ut, cost_improvement

    def train(self, data: torch.Tensor, 
              num_epochs: int, 
              batch_size: int, 
              concepts_known: Optional[torch.Tensor] = None, 
              concepts_unknown: Optional[torch.Tensor] = None,
              lr: float = 2e-4, 
              lr_gamma: float = 0.998,
              p_drop: float = 0.1,
              ot: bool = False):
        """
        Main training loop for the flow matching model.
        """
        if concepts_known is None and concepts_unknown is None and self.c_dim > 0:
            raise ValueError("Model has c_dim > 0, but no concepts were provided for training.")
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        loss_fn = nn.MSELoss()

        print(f"Training FM for {num_epochs} epochs with batch size {batch_size}...")

        # Create placeholders for missing concepts to keep DataLoader structure consistent
        num_data_points = data.shape[0]
        data_tensors = (data,)
        if concepts_known is not None:
            #concepts_known = torch.zeros(num_data_points, self.c_known_dim, device=self.device)
            data_tensors = data_tensors + (concepts_known,)
        if concepts_unknown is not None:
            data_tensors = data_tensors + (concepts_unknown,)
            #concepts_unknown = torch.zeros(num_data_points, self.c_unknown_dim, device=self.device)

        dataset = TensorDataset(*data_tensors)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = self.num_workers)
        
        pbar = tqdm(range(num_epochs), ncols=88)
        self.losses = []

        def unpack_batch(batch):
            x1_batch = batch[0].to(self.device)
            c_known_batch = batch[1].to(self.device) if concepts_known is not None else None
            c_unknown_batch = batch[-1].to(self.device) if concepts_unknown is not None else None
            return x1_batch, c_known_batch, c_unknown_batch

        for epoch in pbar:
            epoch_loss, epoch_ot_improv = 0.0, 0.0
            for batch in dataloader:
                x1_batch, c_known_batch, c_unknown_batch = unpack_batch(batch)

                optimizer.zero_grad()
                t, xt, ut, ot_improv = self._get_train_step_data(x1_batch, ot=ot)
                
                c_parts_masked = []
                if self.c_known_dim > 0:
                    mask = (torch.rand_like(c_known_batch) > p_drop).float()
                    c_parts_masked.append(c_known_batch * mask)
                if self.c_unknown_dim > 0:
                    mask = (torch.rand(c_unknown_batch.shape[0], device=self.device) > p_drop).float()
                    c_parts_masked.append(c_unknown_batch * mask[:, None])
                
                c_total = torch.cat(c_parts_masked, dim=1) if c_parts_masked else torch.zeros(xt.shape[0], 0, device=self.device)
                v_pred = self.model(xt, t, c_total)

                loss = loss_fn(v_pred, ut)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_ot_improv += ot_improv

            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            avg_ot_improv = epoch_ot_improv / len(dataloader)
            self.losses.append(avg_loss)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}" + (f", OT Improv: {avg_ot_improv:.4f}" if ot else ""))

    def _prepare_all_concept_vectors(self,
                                     concepts_known: Optional[torch.Tensor],
                                     concepts_unknown: Optional[torch.Tensor],
                                     negative_concepts_known: Optional[torch.Tensor],
                                     negative_concepts_unknown: Optional[torch.Tensor],
                                     num_samples: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Handles logic for preparing all concept tensors based on shapes and defaults.
        """
        if negative_concepts_known is not None and negative_concepts_unknown is None:
            negative_concepts_unknown = concepts_unknown
        if negative_concepts_unknown is not None and negative_concepts_known is None:
            negative_concepts_known = concepts_known

        all_concepts = [c for c in [concepts_known, concepts_unknown, negative_concepts_known, negative_concepts_unknown] if c is not None]
        batch_sizes = {c.shape[0] for c in all_concepts if c.dim() == 2}
        if len(batch_sizes) > 1:
            raise ValueError(f"Inconsistent batch sizes in 2D concept tensors: {batch_sizes}")
        if batch_sizes:
            num_samples = batch_sizes.pop()

        def broadcast(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None: return None
            if tensor.dim() == 1: return tensor.repeat(num_samples, 1).to(self.device)
            return tensor.to(self.device)

        return broadcast(concepts_known), broadcast(concepts_unknown), broadcast(negative_concepts_known), broadcast(negative_concepts_unknown), num_samples

    @torch.no_grad()
    def sample(self,
             num_samples: int,
             timesteps: int,
             concepts_known: Optional[torch.Tensor] = None,
             concepts_unknown: Optional[torch.Tensor] = None,
             w_pos: float = 2.0,
             negative_concepts_known: Optional[torch.Tensor] = None,
             negative_concepts_unknown: Optional[torch.Tensor] = None,
             w_neg: float = 1.0) -> torch.Tensor:
        """
        Generates samples with optional known/unknown and positive/negative guidance.
        """
        self.model.eval()

        c_known_cond, c_unknown_cond, c_known_neg, c_unknown_neg, num_samples = \
            self._prepare_all_concept_vectors(
                concepts_known, concepts_unknown,
                negative_concepts_known, negative_concepts_unknown, num_samples
            )

        c_cond_parts = [c for c in [c_known_cond, c_unknown_cond] if c is not None]
        c_cond = torch.cat(c_cond_parts, dim=1) if c_cond_parts else torch.zeros(num_samples, 0, device=self.device)
        
        c_uncond = torch.zeros_like(c_cond).to(self.device)
        
        c_neg = None
        c_neg_parts = [c for c in [c_known_neg, c_unknown_neg] if c is not None]
        if len(c_neg_parts) == 2: # Both must be present after defaulting
             c_neg = torch.cat(c_neg_parts, dim=1)
        
        xt = torch.randn(num_samples, self.x_dim, device=self.device)
        print(f"Generating {num_samples} samples with CFG, w_pos={w_pos}" + (f" and w_neg={w_neg}" if c_neg is not None else ""))

        dt = 1.0 / timesteps
        for t_step in tqdm(range(timesteps), desc="Sampling", ncols=88):
            t = torch.full((num_samples,), t_step * dt, device=self.device)
            
            v_cond = self.model(xt, t, c_cond)
            v_uncond = self.model(xt, t, c_uncond)
            
            v_guided = v_uncond + w_pos * (v_cond - v_uncond)
            
            if c_neg is not None:
                v_neg = self.model(xt, t, c_neg)
                v_guided +=  (1 - w_neg) * v_neg

            xt = xt + v_guided * dt
            
        return xt
    
    def _prepare_edit_tensors(self,
                             x_orig: torch.Tensor,
                             orig_concepts_known: Optional[torch.Tensor],
                             orig_concepts_unknown: Optional[torch.Tensor],
                             cf_concepts_known: Optional[torch.Tensor],
                             cf_concepts_unknown: Optional[torch.Tensor]
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Handles dimension validation, broadcasting, and combination of tensors for the edit function.
        """
        xt = x_orig.to(self.device)

        # --- Dimension Handling and Validation ---
        if xt.dim() == 1:
            cf_batch_sizes = {c.shape[0] for c in [cf_concepts_known, cf_concepts_unknown] if c is not None and c.dim() == 2}
            if len(cf_batch_sizes) > 1:
                raise ValueError(f"Inconsistent batch sizes in counterfactual concepts: {cf_batch_sizes}")
            
            num_samples = cf_batch_sizes.pop() if cf_batch_sizes else 1
            xt = xt.unsqueeze(0).repeat(num_samples, 1)

        elif xt.dim() == 2:
            num_samples = xt.shape[0]
        else:
            raise ValueError(f"x_orig must be a 1D or 2D tensor, but got {xt.dim()} dimensions.")

        all_concepts = [orig_concepts_known, orig_concepts_unknown, cf_concepts_known, cf_concepts_unknown]
        processed_concepts = []
        for concept in all_concepts:
            if concept is None:
                processed_concepts.append(None)
                continue
            
            concept = concept.to(self.device)
            if concept.dim() == 1:
                processed_concepts.append(concept.unsqueeze(0).repeat(num_samples, 1))
            elif concept.dim() == 2:
                assert concept.shape[0] == num_samples, \
                    f"A concept tensor has batch size {concept.shape[0]}, but expected {num_samples} to match x_orig."
                processed_concepts.append(concept)
            else:
                raise ValueError(f"Concept tensors must be 1D or 2D, but got {concept.dim()} dimensions.")
        
        orig_concepts_known, orig_concepts_unknown, cf_concepts_known, cf_concepts_unknown = processed_concepts

        # --- Combine Concepts into Single Tensors ---
        orig_concept_parts = [c for c in [orig_concepts_known, orig_concepts_unknown] if c is not None]
        c_orig = torch.cat(orig_concept_parts, dim=1) if orig_concept_parts else torch.zeros(num_samples, 0, device=self.device)

        cf_concept_parts = [c for c in [cf_concepts_known, cf_concepts_unknown] if c is not None]
        c_cf = torch.cat(cf_concept_parts, dim=1) if cf_concept_parts else torch.zeros(num_samples, 0, device=self.device)

        return xt, c_orig, c_cf, num_samples

    @torch.no_grad()
    def edit(self,
             x_orig: torch.Tensor,
             timesteps: int,
             t_edit: float,
             w_cfg: float = 1.0,
             noise_scale: float = 0.5,
             orig_concepts_known: Optional[torch.Tensor] = None,
             orig_concepts_unknown: Optional[torch.Tensor] = None,
             cf_concepts_known: Optional[torch.Tensor] = None,
             cf_concepts_unknown: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates a counterfactual for a given data point 'x_orig' (at t=1) by changing its conditions.
        It performs partial REVERSE integration (adds noise) with the original concepts, then FORWARD
        integration (denoises) with the counterfactual concepts, with optional Classifier-Free Guidance.

        Args:
            x_orig (torch.Tensor): The original data point(s) to edit (at t=1). Shape (batch_size, x_dim) or (x_dim,).
            timesteps (int): The total number of integration steps for a full solve (t=0 to t=1).
            t_edit (float): A value between 0 and 1 controlling edit strength.
                            A higher value means integrating further back towards noise, resulting in a stronger edit.
            w_cfg (float): Guidance scale for CFG. w_cfg=1.0 means no guidance.
            orig_concepts_known (Optional[torch.Tensor]): Original "known" conditions for x_orig.
            orig_concepts_unknown (Optional[torch.Tensor]): Original "unknown" conditions for x_orig.
            cf_concepts_known (Optional[torch.Tensor]): Counterfactual "known" conditions.
            cf_concepts_unknown (Optional[torch.Tensor]): Counterfactual "unknown" conditions.

        Returns:
            torch.Tensor: The edited, counterfactual data point(s).
        """
        self.model.eval()
        if not (0 <= t_edit < 1):
            raise ValueError("t_edit must be between 0 and 1.")

        # --- Calculate distinct inputs for print statement ---
        num_distinct_x = 1 if x_orig.dim() == 1 else x_orig.shape[0]
        cf_batch_sizes = {c.shape[0] for c in [cf_concepts_known, cf_concepts_unknown] if c is not None and c.dim() == 2}
        num_distinct_cf = cf_batch_sizes.pop() if cf_batch_sizes else 1

        xt, c_orig, c_cf, num_samples = self._prepare_edit_tensors(
            x_orig, orig_concepts_known, orig_concepts_unknown, 
            cf_concepts_known, cf_concepts_unknown
        )
        
        print(f"Generating {num_samples} counterfactual sample(s) from {num_distinct_x} original data point(s) with {num_distinct_cf} counterfactual condition(s).")
        if w_cfg != 1.0:
            print(f"Applying Classifier-Free Guidance with w_cfg={w_cfg}.")

        # --- Integration Process ---
        dt = 1.0 / timesteps
        
        # Snap t_edit to the nearest discrete timestep
        t_edit_step = int(round(t_edit * timesteps))
        t_edit_snapped = t_edit_step / timesteps
        if abs(t_edit - t_edit_snapped) > 1e-6:
            print(f"Warning: t_edit snapped from {t_edit:.4f} to {t_edit_snapped:.4f} to align with discrete timesteps.")

        c_uncond = torch.zeros_like(c_orig)

        # 1. Reverse integration (Encoding / Adding Noise) from t=1 down to t=t_edit
        print(f"Encoding: Integrating backward to t={t_edit_snapped:.2f}...")
        # Loop from t=1 down to t=t_edit+dt
        for t_step in tqdm(range(timesteps, t_edit_step, -1), desc="Encoding (Backward)", ncols=88):
            t_val = t_step / timesteps
            t = torch.full((num_samples,), t_val, device=self.device)
            
            v_cond = self.model(xt, t, c_orig)
            #v_uncond = self.model(xt, t, c_uncond)
            #v = v_uncond + w_cfg * (v_cond - v_uncond) # CFG
            v = v_cond
            
            xt = xt - v * dt # Step from t to t-dt

        # add some noise to xt

        xt = xt + torch.randn_like(xt) * noise_scale

        # 2. Forward integration (Decoding / Denoising) from t=t_edit up to t=1
        print(f"Decoding: Integrating forward to t=1.0 with new condition and noise {noise_scale}...")
        # Loop from t=t_edit up to t=1-dt
        for t_step in tqdm(range(t_edit_step, timesteps), desc="Decoding (Forward)", ncols=88):
            t_val = t_step / timesteps
            t = torch.full((num_samples,), t_val, device=self.device)

            v_cond = self.model(xt, t, c_cf)
            v_uncond = self.model(xt, t, c_uncond)
            v = v_uncond + w_cfg * (v_cond - v_uncond) # CFG

            xt = xt + v * dt # Step from t to t+dt
            
        return xt

class CBMFM_MetaTrainer:
    def __init__(self,
                 fm_mod_cfg,
                 cbm_mod = None,
                 num_epochs = 1000,
                 batch_size = 128,
                 lr = 3e-4,
                 raw = False,
                 concept_key= "concepts",
                 num_workers = 0,
                 obsm_key: str = "X",
                 z_score: bool = False,
                 edit:bool = True):
        """
        Class to train and predict interventions with a scCBMG-FM model
        Inputs:
        - cbm_mod: the scCBMG model underlying the model
        - fm_mod: the Flow Matching model configuration
        - num_epochs: max number of epochs to train for
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key: key of the anndata obsm to train on.
        - zscore: whether to whiten the data
        - raw: whether to use "CellFlow" style - using only raw concepts
        """
        self.scCBGM_model = cbm_mod
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.concept_key = concept_key
        self.raw = raw
        self.num_workers = num_workers
        self.obsm_key = obsm_key
        self.z_score = z_score

        self.fm_mod_cfg = fm_mod_cfg

        self.edit = edit

    def get_learned_concepts(self, adata_full):
        """Uses a trained scCBGM to generate learned concepts for all data."""
        print("Generating learned concepts from scCBGM...")
        
        if(self.obsm_key != 'X'):
            all_x = torch.tensor(adata_full.obsm[self.obsm_key], dtype=torch.float32).to('cuda')
        else:
            all_x = torch.tensor(adata_full.X, dtype=torch.float32).to('cuda')

        if self.scCBGM_model.model is None:
            raise ValueError("scCBGM model is not trained. Call fit_cbm_model first")
       
        with torch.no_grad():
            enc = self.scCBGM_model.model.encode(all_x)
            adata_full.obsm['scCBGM_concepts_known'] = self.scCBGM_model.model.cb_concepts_layers(enc['mu']).cpu().numpy()
            adata_full.obsm['scCBGM_concepts_unknown'] = self.scCBGM_model.model.cb_unk_layers(enc['mu']).cpu().numpy()
        
        adata_full.obsm['scCBGM_concepts'] = np.concatenate([adata_full.obsm['scCBGM_concepts_known'], adata_full.obsm['scCBGM_concepts_unknown']], axis=1)

        return adata_full
    
    def fit_cbm_model(self, adata):
        self.scCBGM_model.train(adata)

    def train(self, adata_train):
        """Trains and returns the CB-FM model using learned concepts."""
        print("Training CB-FM model with learned concepts...")

        if not self.raw:
            self.fit_cbm_model(adata_train)
            adata_train = self.get_learned_concepts(adata_train.copy()) 

        if(self.obsm_key != 'X'):
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            if(self.z_score):
                data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization

        if self.raw:
            mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
            mod_cfg["input_dim"] = data_matrix.shape[1]
            mod_cfg["n_concepts"] =adata_train.obsm[self.concept_key].to_numpy().shape[1] 
            self.fm_model = clab.models.cond_fm.Cond_FM(config=mod_cfg)

            self.fm_model.train_loop(
                data=torch.from_numpy(data_matrix.astype(np.float32)),
                concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
                num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.lr,
                )

        else:
            mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
            mod_cfg["input_dim"] = data_matrix.shape[1]
            mod_cfg["n_concepts"] = adata_train.obsm['scCBGM_concepts'].shape[1]
            self.fm_model = clab.models.cond_fm.Cond_FM(config=mod_cfg)
            
            self.fm_model.train_loop(
                data=torch.from_numpy(data_matrix.astype(np.float32)),
                concepts=torch.from_numpy(adata_train.obsm['scCBGM_concepts'].astype(np.float32)),
                num_epochs=self.num_epochs, 
                batch_size=self.batch_size, 
                lr=self.lr,
                num_workers = self.num_workers
                )
        return self.fm_model

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip, values_to_set = None):
        """Performs intervention using a trained learned-concept CB-FM model.
        Concepts_to_flip: List of binary concepts to flip during intervention.
        
        
        edit: if True uses the encode/decode setup, otherwise just decodes.
        """

        print("Performing intervention with CB-FM (learned)...")

        concept_to_intervene_idx = [idx for idx,c in enumerate(adata_inter.obsm[self.concept_key].columns) if c in concepts_to_flip]

        if not self.raw:
            adata_inter = self.get_learned_concepts(adata_inter.copy())

            c_known_inter = torch.from_numpy(adata_inter.obsm['scCBGM_concepts_known'].astype(np.float32))
            c_unknown_inter = torch.from_numpy(adata_inter.obsm['scCBGM_concepts_unknown'].astype(np.float32))
        
            inter_concepts_known = c_known_inter.clone()
            
            inter_concepts_known[:, concept_to_intervene_idx] = 1 - torch.Tensor(adata_inter.obsm[self.concept_key].values)[:, concept_to_intervene_idx] # Set stim concept to the opposite of the observed value.
            

        if(self.obsm_key != 'X'):
                x_inter = adata_inter.obsm[self.obsm_key]
        else:
            x_inter = adata_inter.X
            if(self.z_score):
                x_inter = (x_inter - adata_inter.var['mean'].to_numpy()[None, :]) / adata_inter.var['std'].to_numpy()[None, :]

        # Using raw concepts
        if self.raw:
            init_concepts = adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32)   
            edit_concepts = init_concepts.copy()
            edit_concepts[:, concept_to_intervene_idx] = 1 - edit_concepts[:, concept_to_intervene_idx] # Set stim concept to the opposite of the observed value.

            if self.edit:
                inter_preds = self.fm_model.edit(
                    x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                    c = torch.from_numpy(init_concepts).to('cuda'),
                    c_prime = torch.from_numpy(edit_concepts).to('cuda'),
                    t_edit = 0.0,
                    n_steps = 1000,
                    w_cfg_forward = 1.0,
                    w_cfg_backward = 1.0,
                    noise_add = 0.0)

            else:
                inter_preds = self.fm_model.decode(
                    h=torch.from_numpy(edit_concepts).to('cuda'),
                    n_steps = 1000,
                    w_cfg = 1.0)
            inter_preds = inter_preds.detach().cpu().numpy()

        # Using CBM concepts
        else:

            init_concepts = np.concatenate([c_known_inter, c_unknown_inter], axis=1)
            edit_concepts = np.concatenate([inter_concepts_known, c_unknown_inter], axis=1)
            
            if self.edit:
                inter_preds = self.fm_model.edit(
                    x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                    c = torch.from_numpy(init_concepts.astype(np.float32)).to('cuda'),
                    c_prime = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                    t_edit = 0.,
                    n_steps = 1000,
                    w_cfg_forward = 1.0,
                    w_cfg_backward = 1.0,
                    noise_add = 0.0)
            else:
                inter_preds = self.fm_model.decode(
                h = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                n_steps = 1000,
                w_cfg = 1.0)
            
            inter_preds = inter_preds.detach().cpu().numpy()

        if(self.obsm_key != 'X'):
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds
            if(self.z_score):
                x_inter_preds = (x_inter_preds * adata_inter.var['std'].to_numpy()[None, :]) + adata_inter.var['mean'].to_numpy()[None, :]


        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs['ident'] = 'intervened on'

        if(self.obsm_key != 'X'):
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata


class Mixed_CBMFM_MetaTrainer:
    def __init__(self,
                 fm_mod_cfg,
                 cbm_mod = None,
                 num_epochs = 1000,
                 batch_size = 128,
                 lr = 3e-4,
                 raw = False,
                 concept_key= "concepts",
                 num_workers = 0,
                 obsm_key: str = "X",
                 z_score: bool = False,
                 edit:bool = True,
                 hard_concept_key: str = None,
                 soft_concept_key: str = None):
        """
        Class to train and predict interventions with a scCBMG-FM model
        Inputs:
        - cbm_mod: the scCBMG model underlying the model
        - fm_mod: the Flow Matching model configuration
        - num_epochs: max number of epochs to train for
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - obsm_key: key of the anndata obsm to train on.
        - zscore: whether to whiten the data
        - raw: whether to use "CellFlow" style - using only raw concepts
        """
        self.scCBGM_model = cbm_mod
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.concept_key = concept_key
        self.raw = raw
        self.num_workers = num_workers
        self.obsm_key = obsm_key
        self.z_score = z_score
        self.hard_concept_key = hard_concept_key
        self.soft_concept_key = soft_concept_key

        self.fm_mod_cfg = fm_mod_cfg

        self.edit = edit

    def get_learned_concepts(self, adata_full):
        """Uses a trained scCBGM to generate learned concepts for all data."""
        print("Generating learned concepts from scCBGM...")
        
        if(self.obsm_key != 'X'):
            all_x = torch.tensor(adata_full.obsm[self.obsm_key], dtype=torch.float32).to('cuda')
        else:
            all_x = torch.tensor(adata_full.X, dtype=torch.float32).to('cuda')
        with torch.no_grad():
            enc = self.scCBGM_model.model.encode(all_x)
            if(self.scCBGM_model.model.use_hard_concepts):
                scCBGM_concepts_known_hard = self.scCBGM_model.model.cb_hard_layers(enc['mu']).cpu().numpy()
                scCBGM_concepts_known_hard_df = pd.DataFrame(scCBGM_concepts_known_hard, 
                                                         index=adata_full.obs.index, 
                                                         columns=adata_full.obsm[self.hard_concept_key].columns)
            if(self.scCBGM_model.model.use_soft_concepts):
                scCBGM_concepts_known_soft= self.scCBGM_model.model.cb_soft_layers(enc['mu']).cpu().numpy()
                scCBGM_concepts_known_soft_df = pd.DataFrame(scCBGM_concepts_known_soft,
                                                         index=adata_full.obs.index,
                                                         columns=adata_full.obsm[self.soft_concept_key].columns)

            scCBGM_concepts_unknown = self.scCBGM_model.model.cb_unk_layers(enc['mu']).cpu().numpy()
            scCBGM_concepts_unknown_df = pd.DataFrame(scCBGM_concepts_unknown, 
                                                    index=adata_full.obs.index, 
                                                    columns=[f'unknown_{i}' for i in range(scCBGM_concepts_unknown.shape[1])])

        if(self.scCBGM_model.model.use_hard_concepts and self.scCBGM_model.model.use_soft_concepts):
            scCBGM_concepts = pd.concat([scCBGM_concepts_known_hard_df, scCBGM_concepts_known_soft_df, scCBGM_concepts_unknown_df], axis=1)
        elif(self.scCBGM_model.model.use_hard_concepts):
            scCBGM_concepts = pd.concat([scCBGM_concepts_known_hard_df, scCBGM_concepts_unknown_df], axis=1)
        elif(self.scCBGM_model. model.use_soft_concepts):
            scCBGM_concepts = pd.concat([scCBGM_concepts_known_soft_df, scCBGM_concepts_unknown_df], axis=1)
        else:
            raise ValueError("Model has no known concepts to extract.")
    
        adata_full.obsm['scCBGM_concepts'] = scCBGM_concepts
        return adata_full
    
    def fit_cbm_model(self, adata):
        self.scCBGM_model.train(adata)

    def train(self, adata_train):
        """Trains and returns the CB-FM model using learned concepts."""
        print("Training CB-FM model with learned concepts...")

        if not self.raw:
            self.fit_cbm_model(adata_train)
            adata_train = self.get_learned_concepts(adata_train.copy()) 

        if(self.obsm_key != 'X'):
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            if(self.z_score):
                data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization

        if self.raw:
            mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
            mod_cfg["input_dim"] = data_matrix.shape[1]
            mod_cfg["n_concepts"] =adata_train.obsm[self.concept_key].to_numpy().shape[1] 
            self.fm_model = clab.models.cond_fm.Cond_FM(config=mod_cfg)

            self.fm_model.train_loop(
                data=torch.from_numpy(data_matrix.astype(np.float32)),
                concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
                num_epochs=self.num_epochs, batch_size=self.batch_size, lr=self.lr,
                )

        else:
            mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
            mod_cfg["input_dim"] = data_matrix.shape[1]
            mod_cfg["n_concepts"] = adata_train.obsm['scCBGM_concepts'].shape[1]
            self.fm_model = clab.models.cond_fm.Cond_FM(config=mod_cfg)
            
            self.fm_model.train_loop(
                data=torch.from_numpy(data_matrix.astype(np.float32)),
                concepts=torch.from_numpy(adata_train.obsm['scCBGM_concepts'].to_numpy().astype(np.float32)),
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                num_workers=self.num_workers
                )
        return self.fm_model

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip, values_to_set = None):
        """Performs intervention using a trained learned-concept CB-FM model.
        Concepts_to_flip: List of binary concepts to flip during intervention.
        
        
        edit: if True uses the encode/decode setup, otherwise just decodes.
        """

        print("Performing intervention with CB-FM (learned)...")

        concept_to_intervene_idx = [idx for idx,c in enumerate(adata_inter.obsm[self.concept_key].columns) if c in concepts_to_flip]

        if not self.raw:
            adata_inter = self.get_learned_concepts(adata_inter.copy())

            if(self.obsm_key != 'X'):
                x_inter = adata_inter.obsm[self.obsm_key]
            else:
                x_inter = adata_inter.X

            init_concepts = adata_inter.obsm["scCBGM_concepts"]
            edit_concepts = init_concepts.copy()

            for concept_to_flip, value_to_set in zip(concepts_to_flip, values_to_set):
                edit_concepts[concept_to_flip] = value_to_set
            # edit_concepts[:, -1] = 1 # Set stim concept to 1

            init_concepts = init_concepts.to_numpy().astype(np.float32)
            edit_concepts = edit_concepts.to_numpy().astype(np.float32)


        # Using raw concepts
        if self.raw:
            if(self.obsm_key != 'X'):
                x_inter = adata_inter.obsm[self.obsm_key]
            else:
                x_inter = adata_inter.X
    
            init_concepts = adata_inter.obsm[self.concept_key] 
            edit_concepts = init_concepts.copy()
            for concept_to_flip, value_to_set in zip(concepts_to_flip, values_to_set):
                edit_concepts[concept_to_flip] = value_to_set

            init_concepts = init_concepts.to_numpy().astype(np.float32)
            edit_concepts = edit_concepts.to_numpy().astype(np.float32)

            if self.edit:
                inter_preds = self.fm_model.edit(
                    x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                    c = torch.from_numpy(init_concepts).to('cuda'),
                    c_prime = torch.from_numpy(edit_concepts).to('cuda'),
                    t_edit = 0.0,
                    n_steps = 1000,
                    w_cfg_forward = 1.0,
                    w_cfg_backward = 1.0,
                    noise_add = 0.0)

            else:
                inter_preds = self.fm_model.decode(
                    h=torch.from_numpy(edit_concepts).to('cuda'),
                    n_steps = 1000,
                    w_cfg = 1.0)
            inter_preds = inter_preds.detach().cpu().numpy()

        # Using CBM concepts
        else:
            if self.edit:
                inter_preds = self.fm_model.edit(
                    x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                    c = torch.from_numpy(init_concepts.astype(np.float32)).to('cuda'),
                    c_prime = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                    t_edit = 0.,
                    n_steps = 1000,
                    w_cfg_forward = 1.0,
                    w_cfg_backward = 1.0,
                    noise_add = 0.0)
            else:
                inter_preds = self.fm_model.decode(
                    h = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                    n_steps = 1000,
                    w_cfg = 1.0)
            
            inter_preds = inter_preds.detach().cpu().numpy()

        if(self.obsm_key != 'X'):
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds
            if(self.z_score):
                x_inter_preds = (x_inter_preds * adata_inter.var['std'].to_numpy()[None, :]) + adata_inter.var['mean'].to_numpy()[None, :]


        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs['ident'] = 'intervened on'

        if(self.obsm_key != 'X'):
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata
    

class ConceptFlow_MetaTrainer:
    def __init__(self,
                 mod_cfg,
                 num_epochs = 1000,
                 batch_size = 128,
                 lr = 3e-4,
                 raw = False,
                 concept_key= "concepts",
                 num_workers = 0,
                obsm_key: str = "X",
                 z_score: bool = False):
        """
        Class to train and predict interventions with a scCBMG-FM model
        Inputs:
        - cbm_mod: the scCBMG model underlying the model
        - fm_mod: the Flow Matching model configuration
        - num_epochs: max number of epochs to train for
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - pca: whether to train, and predict in PCA space.
        - zscore: whether to whiten the data
        - raw: whether to use "CellFlow" style - using only raw concepts
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.concept_key = concept_key
        self.raw = raw
        self.num_workers = num_workers
        self.obsm_key = obsm_key
        self.z_score = z_score

        self.mod_cfg = mod_cfg
    
    def train(self, adata_train):
        """Trains and returns the CB-FM model using learned concepts."""
        print("Training CB-FM model with learned concepts...")

        if(self.obsm_key != 'X'):
            data_matrix = adata_train.obsm[self.obsm_key]
        else:
            data_matrix = adata_train.X
            if(self.z_score):
                data_matrix = (data_matrix - adata_train.var['mean'].to_numpy()[None, :]) / adata_train.var['std'].to_numpy()[None, :]  # Z-score normalization

        mod_cfg = hydra.utils.instantiate(self.mod_cfg)
        mod_cfg["input_dim"] = data_matrix.shape[1]
        mod_cfg["n_concepts"] = adata_train.obsm[self.concept_key].to_numpy().shape[1]
        
        self.fm_model = clab.models.concept_fm.Concept_FM(config=mod_cfg)
        self.fm_model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(adata_train.obsm[self.concept_key].to_numpy().astype(np.float32)),
            num_epochs=self.num_epochs, 
            batch_size=self.batch_size, lr=self.lr,
            num_workers = self.num_workers
            )
        return self.fm_model

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip):
        """Performs intervention using a trained learned-concept CB-FM model.
        Concepts_to_flip: List of binary concepts to flip during intervention."""

        print("Performing intervention with CB-FM (learned)...")
        if(self.obsm_key != 'X'):
            x_intervene_on =  torch.tensor(adata_inter.obsm[self.obsm_key], dtype=torch.float32)
        else:
            x_intervene_on = torch.tensor(adata_inter.X, dtype=torch.float32)
        c_intervene_on = adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32)

        concept_to_intervene_idx = [idx for idx,c in enumerate(adata_inter.obsm[self.concept_key].columns) if c in concepts_to_flip]

        #x_intervene_on = torch.from_numpy(data_matrix)
        #c_intervene_on = torch.from_numpy(adata_inter.obsm[self.concept_key].to_numpy().astype(np.float32))

        #inter_concepts_known = c_intervene_on.clone()
        #inter_concepts_known[:, concept_to_intervene_idx] = 1 - torch.Tensor(adata_inter.obsm[self.concept_key].values)[:, concept_to_intervene_idx] # Set stim concept to the opposite of the observed value.

        mask = torch.zeros(c_intervene_on.shape, dtype=torch.float32)
        mask[:, concept_to_intervene_idx] = 1

        inter_concepts = torch.tensor(c_intervene_on, dtype=torch.float32)
        inter_concepts[:, concept_to_intervene_idx] = 1 - inter_concepts[:, concept_to_intervene_idx] # Set stim concept to 1

        #inter_concepts = torch.tensor(c_intervene_on, dtype=torch.float32)
        #inter_concepts[:, -1] = 1 # Set stim concept to 1

        with torch.no_grad():
            inter_preds = self.fm_model.intervene(x_intervene_on.to('cuda'), mask=mask.to('cuda'), concepts=inter_concepts.to('cuda'))

        inter_preds = inter_preds.detach().cpu().numpy()

        if(self.obsm_key != 'X'):
            x_inter_preds = np.zeros_like(adata_inter.X)
        else:
            x_inter_preds = inter_preds
            if(self.z_score):
                x_inter_preds = (x_inter_preds * adata_inter.var['std'].to_numpy()[None, :]) + adata_inter.var['mean'].to_numpy()[None, :]

        pred_adata = adata_inter.copy()
        pred_adata.X = x_inter_preds
        pred_adata.obs['ident'] = 'intervened on'

        if(self.obsm_key != 'X'):
            pred_adata.obsm[self.obsm_key] = inter_preds

        return pred_adata