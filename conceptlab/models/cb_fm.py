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
        print(f"Generating {num_samples} samples with w_pos={w_pos}" + (f" and w_neg={w_neg}" if c_neg is not None else ""))

        dt = 1.0 / timesteps
        for t_step in tqdm(range(timesteps), desc="Sampling", ncols=88):
            t = torch.full((num_samples,), t_step * dt, device=self.device)
            
            v_cond = self.model(xt, t, c_cond)
            v_uncond = self.model(xt, t, c_uncond)
            
            v_guided = v_uncond + w_pos * (v_cond - v_uncond)
            
            if c_neg is not None:
                v_neg = self.model(xt, t, c_neg)
                v_guided += w_neg * (v_cond - v_neg)
            
            xt = xt + v_guided * dt
            
        return xt