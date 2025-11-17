import torch.nn as nn
import torch.nn.functional as F
import torch as t
import torch

import math
from tqdm import tqdm

from typing import Optional


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.
        Args:
            x (torch.Tensor): The main input tensor from the previous layer.
            c_emb (torch.Tensor): The embedding of the concept vector.
            t_emb (torch.Tensor): The embedding of the time value.
        Returns:
            torch.Tensor: The output tensor of the block.
        """
        residual = x
        # Add concept and time embeddings to the main data path before the linear layer
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual

class DecoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_unknown, hidden_dim)
        self.layers = nn.ModuleList(
            [MLPBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim )

    def forward(self, input_concept, unknown, **kwargs):
        x = torch.concat((unknown, input_concept), dim=1)
        
        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_later(h)

        return dict(x_pred=h)
    


class SkipBlock(nn.Module):
    """
    A single block of the MLP, consisting of a linear layer, normalization,
    activation, and dropout, with skip connections for the main data path.
    """
    def __init__(self, emb_dim: int, c_emb_dim = int, dropout: float = 0.1):
        """
        Initializes the MLP block.
        Args:
            emb_dim (int): The dimension of the input, output, and embeddings.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.linear = nn.Linear(emb_dim + c_emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.
        Args:
            x (torch.Tensor): The main input tensor from the previous layer.
            c_emb (torch.Tensor): The embedding of the concept vector.
            t_emb (torch.Tensor): The embedding of the time value.
        Returns:
            torch.Tensor: The output tensor of the block.
        """
        residual = x
        # Add concept embeddings to the main data path before the linear layer
        x = self.linear(torch.concat((x, c_emb), dim=-1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual
    

class SkipDecoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.concept_embedder = nn.Linear(self.n_concepts, hidden_dim)
        self.unknown_embedder = nn.Linear(self.n_unknown, hidden_dim)
        
        self.layers = nn.ModuleList(
            [SkipBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, input_concept, unknown, **kwargs):
        c = self.concept_embedder(input_concept)
        h = self.unknown_embedder(unknown)

        for layer in self.layers:
            h = layer(h, c)

        h = self.output_later(h)
        return dict(x_pred=h)
    





class NoResBlock(nn.Module):
    """
    A single block of the MLP, consisting of a linear layer, normalization,
    activation, and dropout, without skip connections.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.
        Args:
            x (torch.Tensor): The main input tensor from the previous layer.
            c_emb (torch.Tensor): The embedding of the concept vector.
            t_emb (torch.Tensor): The embedding of the time value.
        Returns:
            torch.Tensor: The output tensor of the block.
        """
        # Add concept and time embeddings to the main data path before the linear layer
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class NoResDecoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_unknown, hidden_dim)
        self.layers = nn.ModuleList(
            [NoResBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, input_concept, unknown, **kwargs):
        x = torch.concat((unknown, input_concept), dim=1)
        
        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_later(h)

        return dict(x_pred=h)
    
class CVAEDecoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_latent: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_latent = n_latent
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_latent, hidden_dim)
        self.layers = nn.ModuleList(
            [NoResBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, latent, input_concept, **kwargs):
        x = torch.concat((latent, input_concept), dim=1)
        
        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_later(h)

        return dict(x_pred=h)
    


class DefaultDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_unknown: int,
        hidden_dim: int,
        dropout: float,
        n_concepts: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim]
        )
        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        layers = []
        layers_dim = [self.n_concepts + self.n_unknown] + self.hidden_dim

        for k in range(0, len(layers_dim) - 1):

            layer_k = [
                nn.Linear(layers_dim[k], layers_dim[k + 1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]

            layers += layer_k

        layers.append(nn.Linear(self.hidden_dim[-1], self.input_dim))

        if len(self.hidden_dim) == 1 and (self.hidden_dim[0] == self.input_dim):
            layers.append(nn.ReLU())

        self.decoder_layers = nn.Sequential(*layers)

    def forward(self, h, **kwargs):
        h = self.decoder_layers(h)
        return dict(x_pred=h)


class ConditionalDecoderBlock(DefaultDecoderBlock):
    def forward(self, h, concepts=None, **kwargs):
        h0 = t.cat((h, concepts), dim=1)
        h = self.decoder_layers(h0)

        return dict(x_pred=h)


# class SkipLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, dropout_p=0.0, is_last_layer=False):
#         super(SkipLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, hidden_dim)
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.is_last_layer = is_last_layer

#     def forward(self, inputs):
#         h, c = inputs
#         x = t.cat((h, c), dim=-1)
#         x = self.fc(x)
#         if not self.is_last_layer:
#             x = F.relu(x)
#             x = self.dropout(x)
#         return x, c


# class SkipDecoderBlock(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         n_unknown: int,
#         hidden_dim: int,
#         dropout: float = 0.0,
#         n_concepts: int = 0,
#         **kwargs,
#     ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = (
#             hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim]
#         )
#         self.n_concepts = n_concepts
#         self.n_unknown = n_unknown
#         self.dropout = dropout

#         layers = []
#         layers_dim = [self.n_concepts + self.n_unknown] + self.hidden_dim

#         for k in range(0, len(layers_dim) - 1):

#             layers.append(
#                 SkipLayer(
#                     layers_dim[k],
#                     layers_dim[k + 1],
#                     self.dropout,
#                     is_last_layer=False,
#                 )
#             )

#         layers.append(
#             SkipLayer(
#                 self.hidden_dim[-1] + self.n_concepts,
#                 self.input_dim,
#                 is_last_layer=True,
#             )
#         )

#         if len(self.hidden_dim) == 1 and (self.hidden_dim[0] == self.input_dim):
#             layers.append(nn.ReLU())
#         self.decoder_layers = nn.Sequential(*layers)

#     def forward(self, input_concept, unknown, **kwargs):
#         h, _ = self.decoder_layers((unknown, input_concept))
#         return dict(x_pred=h)

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


class FlowMLPBlock(nn.Module):
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
        residual = x
        # Add concept and time embeddings to the main data path before the linear layer
        x = self.linear(x + c_emb + t_emb)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual

class FlowDecoder(nn.Module):
    """
    A fully-connected neural network with skip connections, designed for flow matching.
    Can be used for both conditional and unconditional generation.
    """
    def __init__(self, x_dim: int, c_dim: int = 0, emb_dim: int = 256, n_layers: int = 8, dropout: float = 0.1):
        """
        Args:
            x_dim (int): Dimension of the data x.
            c_dim (int): Dimension of the condition c. Set to 0 for an unconditional model.
            emb_dim (int): Dimension of the internal embeddings.
            n_layers (int): Number of FlowMLPBlock layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.emb_dim = emb_dim

        self.x_embedder = nn.Linear(x_dim, emb_dim)
        # c_embedder is only created if c_dim > 0, making the model conditional
        self.c_embedder = nn.Linear(c_dim, emb_dim) if self.c_dim > 0 else None
        self.t_embedder = FourierEmbedding(emb_dim)
        self.t_mapper = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

        self.layers = nn.ModuleList([FlowMLPBlock(emb_dim, dropout) for _ in range(n_layers)])
        self.output_layer = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, self.x_dim))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the velocity field v(x_t, t, c).
        Args:
            x_t (torch.Tensor): The input tensor at time t, shape (batch, x_dim).
            t (torch.Tensor): The time tensor, shape (batch,).
            c (Optional[torch.Tensor]): The conditional tensor, shape (batch, c_dim).
                                         If None, performs unconditional forward pass.
        Returns:
            torch.Tensor: The predicted velocity, shape (batch, x_dim).
        """
        x_emb = self.x_embedder(x_t)

        # Handle optional conditioning
        if c is not None:
            if self.c_embedder is None:
                raise ValueError("Conditional tensor 'c' was provided, but the model was initialized with c_dim=0.")
            c_emb = self.c_embedder(c)
        else:
            # For unconditional generation, use a zero-tensor for the condition embedding
            c_emb = torch.zeros(x_t.shape[0], self.emb_dim, device=x_t.device)

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_fourier_emb = self.t_embedder(t)
        t_emb = self.t_mapper(t_fourier_emb)

        h = x_emb
        for layer in self.layers:
            h = layer(h, c_emb, t_emb)
        output = self.output_layer(h)
        return output

    @torch.no_grad()
    def integrate(self, c: Optional[torch.Tensor] = None, batch_size: Optional[int] = None, x0: Optional[torch.Tensor] = None, steps: int = 100) -> dict:
        """
        Generates samples by integrating from an initial tensor x0 (noise) to data (x1).
        Args:
            c (Optional[torch.Tensor]): The condition for generation. If None, generates unconditionally.
            batch_size (Optional[int]): The number of samples to generate.
                                        Required if 'c' and 'x0' are None. Ignored if 'x0' is provided.
            x0 (Optional[torch.Tensor]): An initial tensor to start the integration from. If None,
                                        a standard normal noise tensor is created.
            steps (int): The number of integration steps.
        Returns:
            dict: A dictionary containing the generated samples {'x_pred': tensor}.
        """
        # --- Parameter Validation ---
        if x0 is None and c is None and batch_size is None:
            raise ValueError("Either 'x0', 'c', or 'batch_size' must be provided.")
        if c is not None and self.c_embedder is None:
            raise ValueError("Conditional tensor 'c' was provided, but the model is unconditional.")
        if x0 is not None and c is not None and x0.size(0) != c.size(0):
            raise ValueError(f"Batch size mismatch: x0 has batch size {x0.size(0)} but c has {c.size(0)}.")

        # --- Determine Generation Parameters ---
        if x0 is not None:
            # Use the provided x0 tensor
            effective_batch_size = x0.size(0)
            device = x0.device
            initial_x = x0.clone().to(device) # Use a clone on the correct device
        elif c is not None:
            # Infer from condition c
            effective_batch_size = c.size(0)
            device = c.device
            initial_x = torch.randn(effective_batch_size, self.x_dim, device=device)
        else:
            # Infer from batch_size argument
            effective_batch_size = batch_size
            device = next(self.parameters()).device
            initial_x = torch.randn(effective_batch_size, self.x_dim, device=device)

        xt = initial_x
        dt = 1.0 / steps

        # --- Run Integration Loop ---
        for t_step in tqdm(range(steps), desc="Forward Process", ncols=88, leave=False):
            t_current = t_step * dt
            t_vec = torch.full((effective_batch_size,), t_current, device=device)

            velocity = self.forward(xt, t_vec, c)
            xt = xt + velocity * dt

        return dict(x_pred=xt)








class NoResDecoderBlock_CEM(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_unknown, hidden_dim)
        self.layers = nn.ModuleList(
            [NoResBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, decoder_input, **kwargs):
        x = decoder_input
        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_later(h)

        return dict(x_pred=h)
    
class SkipDecoderBlock_CEM(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_unknown, hidden_dim)
        
        self.layers = nn.ModuleList(
            [SkipBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, decoder_input, **kwargs):
        x = decoder_input
        h = self.x_embedder(x)

        for layer in self.layers:
            h = layer(h)

        h = self.output_later(h)
        return dict(x_pred=h)
    

class DecoderBlock_CEM(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_concepts: int,
    n_unknown: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.n_concepts + self.n_unknown, hidden_dim)
        self.layers = nn.ModuleList(
            [MLPBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.output_later = nn.Linear(self.hidden_dim, self.input_dim )

    def forward(self, decoder_input, **kwargs):
        x = decoder_input
        
        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_later(h)

        return dict(x_pred=h)