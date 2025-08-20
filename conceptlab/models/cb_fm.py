import torch
import torch.nn as nn
import torch.optim.lr_scheduler 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Tuple


# Assuming the MLP model is defined in mlp.py as specified previously
from .mlp import MLP
from .utils import optimal_transport_coupling


class CB_FM:
    """
    Implements a conditional flow matching model using a base MLP.
    This class handles the training, sampling, and the core logic for
    classifier-free guidance with per-concept dropout.
    """

    def __init__(self, x_dim: int, c_dim: int, emb_dim: int = 256, n_layers: int = 4, dropout: float = 0.1):
        """
        Initializes the FlowMatchingModel.
        Args:
            x_dim (int): The dimensionality of the input data (e.g., cell features).
            c_dim (int): The dimensionality of the conditioning concept vector.
            emb_dim (int): The embedding dimension for the MLP.
            n_layers (int): The number of layers in the MLP.
            dropout (float): The dropout rate for the MLP.
        """
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Instantiate the neural network that learns the vector field
        self.model = MLP(
            x_dim=x_dim,
            c_dim=c_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)

    def _get_train_step_data(self, x1: torch.Tensor, ot: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the required tensors for a single training step based on the flow matching formula.
        
        Args:
            x1 (torch.Tensor): A batch of real data points (target).
        
        Returns:
            A tuple containing:
            - t (torch.Tensor): A batch of random time steps.
            - xt (torch.Tensor): The interpolated data points at time t.
            - ut (torch.Tensor): The target vector field.
        """
        # 1. Sample noise x0 from a standard normal distribution
        x0 = torch.randn_like(x1)

        if ot:
            x0, cost_improvement = optimal_transport_coupling(x0, x1)
        else:
            cost_improvement = 1
        # 2. Sample time t uniformly from [0, 1]
        t = torch.rand(x1.shape[0], device=self.device)
        
        # Reshape t for broadcasting: (batch_size,) -> (batch_size, 1, 1, ...)
        t_reshaped = t.view(-1, *([1] * (x1.dim() - 1)))
        
        # 3. Create the interpolated sample xt
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1
        
        # 4. Define the target vector field u_t
        ut = x1 - x0
        
        return t, xt, ut, cost_improvement

    def train(self, data: torch.Tensor, 
                    concepts: torch.Tensor, 
                    num_epochs: int, 
                    batch_size: int, 
                    lr: float = 2e-4, 
                    lr_gamma: float = 0.998,
                    p_drop: float = 0.1,
                    ot: bool = False):
        """
        Main training loop for the flow matching model.
        
        Args:
            data (torch.Tensor): The training data (e.g., cell embeddings).
            concepts (torch.Tensor): The corresponding concept vectors for the data.
            num_epochs (int): The number of epochs to train for.
            batch_size (int): The size of each training batch.
            lr (float): The learning rate for the Adam optimizer.
            p_drop (float): The probability of dropping a single concept during training for CFG.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_gamma)
        loss_fn = nn.MSELoss()

        print(f"Training FM for {num_epochs} epochs, with a batch size of {batch_size} and ot {ot}, using the adam optimizer with lr = {lr} and {lr_gamma} decay")

        dataset = TensorDataset(data, concepts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        pbar = tqdm(range(num_epochs), ncols=88)
        self.losses = []

        for epoch in pbar:
            epoch_loss = 0.0
            epoch_ot_improv = 0.0
            for x1_batch, c_batch in dataloader:
                x1_batch = x1_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Get the interpolated data and target vector field
                t, xt, ut, ot_improv = self._get_train_step_data(x1_batch, ot = ot)
                
                # --- Per-Concept Dropout for Classifier-Free Guidance ---
                # Create a random binary mask for the concepts
                mask = (torch.rand_like(c_batch) > p_drop).float()
                c_masked = c_batch * mask
                
                # Predict the vector field
                v_pred = self.model(xt, c_masked, t)
                
                # Calculate loss
                loss = loss_fn(v_pred, ut)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_ot_improv += ot_improv

            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            avg_ot_improv = epoch_ot_improv / len(dataloader)
            self.losses.append(avg_loss)
            if(ot):
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}, OT Improvement {avg_ot_improv:.4f}")
            else:
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def sample(self, concepts: torch.Tensor, num_samples: int, timesteps: int, w: float = 2.0) -> torch.Tensor:
        """
        Generates new samples conditioned on the provided concepts using Euler discretization.
        
        Args:
            concepts (torch.Tensor): The concept vector to condition the generation on. Shape (c_dim,).
            num_samples (int): The number of samples to generate.
            timesteps (int): The number of steps for the Euler solver.
            w (float): The guidance scale for Classifier-Free Guidance. w=0 is unconditional.
        
        Returns:
            torch.Tensor: The generated samples.
        """
        self.model.eval()
        
        # Start with random noise
       
        
        # Prepare concept vectors for batch processing'

        if(concepts.dim() == 1):
            c_cond = concepts.repeat(num_samples, 1).to(self.device)
        else:
            num_samples = concepts.shape[0]
            c_cond = concepts.to(self.device)

        c_uncond = torch.zeros_like(c_cond).to(self.device)
        xt = torch.randn(num_samples, self.x_dim, device=self.device)

        print(f"Generating {num_samples} samples with CFG of gamma = {w}")

        dt = 1.0 / timesteps
        

        for t_step in tqdm(range(timesteps), desc="Sampling", ncols=88):
            t = torch.full((num_samples,), t_step * dt, device=self.device)
            
            

            # Predict conditional and unconditional vector fields
            v_cond = self.model(xt, c_cond, t)
            v_uncond = self.model(xt, c_uncond, t)
            
            # Apply classifier-free guidance
            v_guided = v_uncond + w * (v_cond - v_uncond)
            
            # Update the sample using the Euler step
            xt = xt + v_guided * dt
            
        return xt
