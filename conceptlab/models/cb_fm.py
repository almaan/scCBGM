import torch
import torch.nn as nn
import torch.optim.lr_scheduler 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import List, Optional, Tuple


# Assuming the MLP model is defined in mlp.py as specified previously
from .mlp import MLP
from .utils import optimal_transport_coupling



class CB_FM:
    """
    Implements a conditional flow matching model using a base MLP.
    This class handles training and sampling with optional "known" and "unknown" concepts.
    """

    def __init__(self, x_dim: int, c_known_dim: int = 0, c_unknown_dim: int = 0, emb_dim: int = 256, n_layers: int = 4, dropout: float = 0.1):
        """
        Initializes the FlowMatchingModel.
        Args:
            x_dim (int): The dimensionality of the input data (e.g., cell features).
            c_known_dim (int): The dimensionality of the "known" conditioning vector.
            c_unknown_dim (int): The dimensionality of the "unknown" conditioning vector.
            emb_dim (int): The embedding dimension for the MLP.
            n_layers (int): The number of layers in the MLP.
            dropout (float): The dropout rate for the MLP.
        """
        self.x_dim = x_dim
        self.c_known_dim = c_known_dim
        self.c_unknown_dim = c_unknown_dim
        self.c_dim = c_known_dim + c_unknown_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.c_dim == 0:
            print("Warning: All concept dimensions are 0. Model will be purely unconditional.")

        self.model = MLP(
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
        if concepts_known is None:
            concepts_known = torch.zeros(num_data_points, self.c_known_dim, device=self.device)
        if concepts_unknown is None:
            concepts_unknown = torch.zeros(num_data_points, self.c_unknown_dim, device=self.device)

        dataset = TensorDataset(data, concepts_known, concepts_unknown)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        pbar = tqdm(range(num_epochs), ncols=88)
        self.losses = []

        for epoch in pbar:
            epoch_loss, epoch_ot_improv = 0.0, 0.0
            for x1_batch, c_known_batch, c_unknown_batch in dataloader:
                x1_batch, c_known_batch, c_unknown_batch = \
                    x1_batch.to(self.device), c_known_batch.to(self.device), c_unknown_batch.to(self.device)

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
            v_uncond = self.model(xt, t, c_uncond)
            v = v_uncond + w_cfg * (v_cond - v_uncond) # CFG
            
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