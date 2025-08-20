import torch
import torch.nn as nn
import math

class FourierEmbedding(nn.Module):
    """
    Projects a scalar time value 't' into a higher-dimensional Fourier feature space.
    This allows the model to more easily understand the continuous nature of time.
    """
    def __init__(self, emb_dim: int):
        """
        Initializes the Fourier embedding layer.
        Args:
            emb_dim (int): The dimension of the embedding. Must be an even number.
        """
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but got {emb_dim}")
        
        # A constant factor used in the Fourier feature calculation.
        # This is not a trainable parameter.
        self.register_buffer(
            "weights", torch.randn(emb_dim // 2) * 2 * math.pi
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Transforms the time tensor into Fourier features.
        Args:
            t (torch.Tensor): A tensor of time values, shape (batch_size, 1).
        Returns:
            torch.Tensor: The Fourier features, shape (batch_size, emb_dim).
        """
        # Calculate the arguments for sin and cos functions
        t_proj = t * self.weights[None, :]
        # Concatenate the sin and cos features to form the final embedding
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class MLPBlock(nn.Module):
    """
    A single block of the MLP, consisting of a linear layer, normalization,
    activation, and dropout, with skip connections for the main data path.
    """
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        """
        Initializes the MLP block.
        Args:
            emb_dim (int): The dimension of the input, output, and embeddings.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.
        Args:
            x (torch.Tensor): The main input tensor from the previous layer.
            c_emb (torch.Tensor): The embedding of the concept vector.
            t_emb (torch.Tensor): The embedding of the time value.
        Returns:
            torch.Tensor: The output tensor of the block.
        """
        # The main path with a residual connection
        # We add the concept and time embeddings before the linear layer

        
        residual = x
        x = self.linear(x + c_emb + t_emb)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class MLP(nn.Module):
    """
    A fully-connected neural network with skip connections, designed for
    flow matching. It takes cell data, concepts, and time as input.
    """
    def __init__(self, x_dim: int, c_dim: int, emb_dim: int, n_layers: int, dropout: float = 0.1):
        """
        Initializes the MLP model.
        Args:
            x_dim (int): The dimensionality of the input cell data (x_t).
            c_dim (int): The dimensionality of the concept vector (c_masked).
            emb_dim (int): The core embedding dimension used throughout the network.
            n_layers (int): The number of MLPBlock layers.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.emb_dim = emb_dim

        # --- Initial Embedding Layers ---
        # Embed the input data x_t to the main embedding dimension
        self.x_embedder = nn.Linear(x_dim, emb_dim)
        
        # Embed the concept vector c_masked to the main embedding dimension
        self.c_embedder = nn.Linear(c_dim, emb_dim)
        
        # Fourier feature embedding for the time scalar t
        self.t_embedder = FourierEmbedding(emb_dim)
        
        # An additional embedding layer for time to allow for more complex transformations
        self.t_mapper = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # --- Core Network ---
        # A stack of MLP blocks that form the main body of the network
        self.layers = nn.ModuleList(
            [MLPBlock(emb_dim, dropout) for _ in range(n_layers)]
        )

        # --- Output Layer ---
        # A final layer to project the output back to the original data dimension
        self.output_layer = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, self.x_dim)
        )

    def forward(self, x_t: torch.Tensor, c_masked: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the entire model.
        Args:
            x_t (torch.Tensor): The noisy cell data, shape (batch_size, x_dim).
            c_masked (torch.Tensor): The masked concept vector, shape (batch_size, c_dim).
            t (torch.Tensor): The time values, shape (batch_size,).
        Returns:
            torch.Tensor: The predicted vector field, shape (batch_size, x_dim).
        """
        # 1. Embed all inputs
        x_emb = self.x_embedder(x_t)
        c_emb = self.c_embedder(c_masked)
        
        # Ensure t is in the correct format (batch_size, 1) for the embedder
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_fourier_emb = self.t_embedder(t)
        t_emb = self.t_mapper(t_fourier_emb)

        # 2. Process through MLP blocks
        h = x_emb
        for layer in self.layers:
            h = layer(h, c_emb, t_emb)

        # 3. Project to output dimension
        output = self.output_layer(h)
        return output

