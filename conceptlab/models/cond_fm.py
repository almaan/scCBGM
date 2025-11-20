import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb

from .utils import sigmoid

from .mlp import FlowDecoder, ConditionEmbedding
from tqdm import tqdm

from abc import ABC


class Cond_FM(nn.Module, ABC):
    def __init__(
        self,
        config,
        _decoder: nn.Module = FlowDecoder,
        **kwargs,
    ):

        super(Cond_FM, self).__init__()

        self.input_dim = config.get("input_dim", 1024)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.latent_dim = config.get("latent_dim", 256)
        self.n_concepts = config.get("n_concepts", 10)
        self.n_layers = config.get("n_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        self.p_uncond = config.get("p_uncond", 0.0)

        if self.p_uncond > 0:
            self.condition_embedder = ConditionEmbedding(
                c_dim=self.n_concepts, emb_dim=self.hidden_dim
            )
            self.concept_emb_dim = self.hidden_dim
        else:
            self.condition_embedder = nn.Identity()
            self.concept_emb_dim = self.n_concepts

        self._decoder = _decoder(
            x_dim=self.input_dim,
            c_dim=self.concept_emb_dim,
            emb_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    @property
    def has_concepts(self):
        return True

    @torch.no_grad()
    def decode(self, h, n_steps: int = 1000, w_cfg: float = 1.0, **kwargs):
        """
        Integrates the flow ODE forward from t=0 to t=1 using the Euler
        method, conditioned on the concept vector h.

        Uses classifier-free guidance for generation.

        Args:
            h (torch.Tensor): The conditional concept vectors.
            n_steps (int): The number of integration steps.
            w_cfg (float): The classifier-free guidance scale.
                           w_cfg=0.0 means unconditional generation.
                           w_cfg=1.0 means conditional generation without guidance.
        """
        batch_size, device = h.shape[0], h.device
        x_t = torch.randn(batch_size, self.input_dim, device=device)
        dt = 1.0 / n_steps

        h_emb = self.condition_embedder(h)

        print(f"Decoding with {n_steps} steps and CFG scale w={w_cfg}")
        # Iteratively solve the ODE using the Euler method
        for t_step in tqdm(
            range(n_steps), desc="Forward Process", ncols=88, leave=False
        ):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            # --- 3. MODIFIED: CFG logic for sampling ---
            # Predict the conditional vector field
            v_cond = self._decoder(x_t=x_t, t=t_vec, c=h_emb)

            # Predict the unconditional vector field if guidance is used
            v_uncond = self._decoder(
                x_t=x_t, t=t_vec, c=torch.zeros_like(h_emb, device=device)
            )

            # Combine them using the guidance scale
            v_guided = v_uncond + w_cfg * (v_cond - v_uncond)

            # Take a step using the guided vector field
            x_t = x_t + v_guided * dt

        x_pred = x_t
        return x_pred

    @torch.no_grad()
    def edit(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        c_prime: torch.Tensor,
        t_edit: float,
        n_steps: int = 1000,
        w_cfg_forward: float = 1.0,
        w_cfg_backward: float = 1.0,
        noise_add: float = 0.1,
    ):
        """
        Edits a sample 'x' from its original condition 'c' to a new condition 'c_prime'.

        This is done by first integrating backwards from x (at t=1) to an intermediate
        time t_edit using the original condition c. Then, it integrates forward from
        that point to t=1 using the new condition c_prime.

        Args:
            x (torch.Tensor): The initial data sample to edit.
            c (torch.Tensor): The original condition vector for x.
            c_prime (torch.Tensor): The target condition vector for editing.
            t_edit (float): The time to integrate back to (between 0.0 and 1.0).
            n_steps (int): Total number of integration steps for a full t=0 to t=1 pass.
            w_cfg_forward (float): The CFG scale for the forward (editing) pass.
            w_cfg_backward (float): The CFG scale for the backward (inversion) pass.
        """
        batch_size, device = x.shape[0], x.device
        dt = 1.0 / n_steps

        # --- Snap t_edit to the discrete time grid ---
        n_edit_steps = int(t_edit * n_steps)
        print(
            f"Editing from t=1.0 back to t={n_edit_steps * dt:.2f}, then forward with new condition."
        )

        x_t = x.clone()

        c_emb = self.condition_embedder(c)
        c_prime_emb = self.condition_embedder(c_prime)
        # --- 1. Backward Process (Inversion) ---
        # Integrate from t=1 down to t_edit using the original condition 'c'
        for t_step in tqdm(
            range(n_steps, n_edit_steps, -1),
            desc="Backward Process (Inversion), cfg: {:.2e}".format(w_cfg_backward),
            ncols=88,
            leave=False,
        ):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            # Use the original condition 'c' and backward cfg scale
            v_cond = self._decoder(x_t=x_t, t=t_vec, c=c_emb)
            v_uncond = self._decoder(
                x_t=x_t, t=t_vec, c=torch.zeros_like(c_emb, device=device)
            )

            v_guided = v_uncond + w_cfg_backward * (v_cond - v_uncond)

            # To go backward in time, we subtract the vector field
            x_t = x_t - v_guided * dt

        if noise_add > 0:
            print(
                "Adding noise to the edited sample with std = {:.2e}".format(noise_add)
            )

        x_t = x_t + noise_add * torch.randn_like(x_t)
        # --- 2. Forward Process (Editing) ---
        # Integrate from t_edit up to t=1 using the new condition 'c_prime'
        for t_step in tqdm(
            range(n_edit_steps, n_steps),
            desc="Forward Process (Editing), cfg: {:.2e}".format(w_cfg_forward),
            ncols=88,
            leave=False,
        ):
            t_current = t_step * dt
            t_vec = torch.full((batch_size,), t_current, device=device)

            # Use the new condition 'c_prime' and forward cfg scale
            v_cond = self._decoder(x_t=x_t, t=t_vec, c=c_prime_emb)
            v_uncond = self._decoder(
                x_t=x_t, t=t_vec, c=torch.zeros_like(c_prime_emb, device=device)
            )
            v_guided = v_uncond + w_cfg_forward * (v_cond - v_uncond)

            # To go forward in time, we add the vector field
            x_t = x_t + v_guided * dt

        x_edited = x_t
        return x_edited

    def fm_loss(self, v_pred, v):
        return F.mse_loss(v_pred, v, reduction="mean")

    def loss_function(self, v_pred, v, **kwargs):
        """Calculates the total loss for the Concept Flow Matching model."""
        loss_dict = {}
        flow_loss = self.fm_loss(v_pred, v)
        loss_dict["fm_loss"] = flow_loss
        loss_dict["Total_loss"] = flow_loss
        return loss_dict

    def train_loop(
        self,
        data: torch.Tensor,
        concepts: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        lr: float = 2e-4,
        lr_gamma: float = 0.997,
        num_workers: int = 0,
    ):
        """
        Defines the training loop for the Concept_FM model.

        This function sets up the optimizer, scheduler, and data loaders, and then
        performs optimization steps over the model's parameters using a tqdm progress bar.
        """
        # --- 1. Setup Device, DataLoader, Optimizer, Scheduler ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.to(device)  # Move the model to the correct device

        # Create DataLoader
        dataset = TensorDataset(data, concepts)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Setup Optimizer and Scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        print(f"Starting training on {device} for {num_epochs} epochs...")

        self.train()  # Set the model to training mode
        # --- 2. The Training Loop ---
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)

        for epoch in pbar:
            total_loss = 0.0

            # Iterate over batches
            for x_batch, concepts_batch in data_loader:
                # Move batch data to the correct device
                x_batch = x_batch.to(device)
                concepts_batch = concepts_batch.to(device)

                concepts_batch_emb = self.condition_embedder(concepts_batch)

                uncond_mask = (
                    torch.rand(concepts_batch_emb.shape[0], device=device)
                    < self.p_uncond
                )
                concepts_batch_emb[uncond_mask] = 0

                # 4. Prepare for Flow Matching Loss
                t = torch.rand(x_batch.shape[0], 1, device=device)
                x_0 = torch.randn_like(x_batch)
                x_1 = x_batch
                x_t = (1 - t) * x_0 + t * x_1
                v = x_1 - x_0

                # 5. Get the Model's Prediction
                v_pred = self._decoder(x_t=x_t, t=t.squeeze(-1), c=concepts_batch_emb)

                # 6. Calculate loss
                loss_dict = self.loss_function(
                    v_pred=v_pred,
                    v=v,
                    concepts=concepts_batch_emb,
                )
                loss = loss_dict["Total_loss"]

                # 7. Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # --- End of Epoch ---
            avg_loss = total_loss / len(data_loader)

            # Update the epoch-level progress bar
            pbar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.3e}",
                    "lr": f"{scheduler.get_last_lr()[0]:.5e}",
                }
            )

            # Step the scheduler
            scheduler.step()

        self.eval()  # Set the model to evaluation mode after training
