import torch.nn as nn
import torch.nn.functional as F
import torch as t
import torch

import math
from tqdm import tqdm

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


class SkipLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.0, is_last_layer=False):
        super(SkipLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.is_last_layer = is_last_layer

    def forward(self, inputs):
        h, c = inputs
        x = t.cat((h, c), dim=-1)
        x = self.fc(x)
        if not self.is_last_layer:
            x = F.relu(x)
            x = self.dropout(x)
        return x, c


class SkipDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_unknown: int,
        hidden_dim: int,
        dropout: float = 0.0,
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

            layers.append(
                SkipLayer(
                    layers_dim[k],
                    layers_dim[k + 1],
                    self.dropout,
                    is_last_layer=False,
                )
            )

        layers.append(
            SkipLayer(
                self.hidden_dim[-1] + self.n_concepts,
                self.input_dim,
                is_last_layer=True,
            )
        )

        if len(self.hidden_dim) == 1 and (self.hidden_dim[0] == self.input_dim):
            layers.append(nn.ReLU())
        self.decoder_layers = nn.Sequential(*layers)

    def forward(self, input_concept, unknown, **kwargs):
        h, _ = self.decoder_layers((unknown, input_concept))
        return dict(x_pred=h)

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
    """
    def __init__(self, x_dim: int, c_dim: int, emb_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.emb_dim = emb_dim

        # --- Embedding Layers ---
        # BUG FIX: Added an x0_embedder, as it's required by the conditional flow matching loss function.
        # self.x0_embedder = nn.Sequential(
        #     nn.Linear(c_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, x_dim)
        # )
        self.x_embedder = nn.Linear(x_dim, emb_dim)
        self.c_embedder = nn.Linear(self.c_dim, emb_dim) if self.c_dim > 0 else None
        self.t_embedder = FourierEmbedding(emb_dim)
        self.t_mapper = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

        self.layers = nn.ModuleList([FlowMLPBlock(emb_dim, dropout) for _ in range(n_layers)])
        self.output_layer = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, self.x_dim))

    # BUG FIX: Added get_x0 method required by the loss function.
    # def get_x0(self, c: torch.Tensor) -> torch.Tensor:
    #     """ Get the initial state of the flow from the bottleneck vector c. """
    #     return self.x0_embedder(c)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_emb = self.x_embedder(x_t)
        
        c_emb = self.c_embedder(c) if self.c_embedder is not None else torch.zeros(x_t.shape[0], self.emb_dim, device=x_t.device)
        
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
    def integrate(self, h: torch.Tensor, steps: int = 100) -> dict:
        """
        Generates x1 by integrating from x0 using the learned velocity field.
        """
        device = h.device
        x0 = torch.randn(h.size(0), self.x_dim, device=device)
        #x0 = torch.zeros(h.size(0), self.x_dim, device=device)
        
        xt = x0.clone()
        dt = 1.0 / steps
        
        for t_step in tqdm(range(steps), desc="Forward Process", ncols=88, leave=False):
            # Call forward with (x, t, c)
            t_current = t_step * dt
            t_vec = torch.full((h.size(0),), t_current, device=device)

            velocity = self.forward(xt, t_vec, h)
            xt = xt + velocity * dt
            
        return dict(x_pred=xt)