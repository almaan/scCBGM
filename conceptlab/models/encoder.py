import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch as t

import torch

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

class NoResMLPBlock(nn.Module):
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
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class NoResEncoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_layers: int,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    n_concepts: int = 0,
    variational: bool = True,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim + n_concepts
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.variational = variational

        self.x_embedder = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [NoResMLPBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)

        if self.variational:
            self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        else:
            self.fc_var = None

    def forward(self, x, concepts=None, **kwargs):

        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h,)

        mu = self.fc_mu(h)
        if self.variational:
            logvar = self.fc_var(h)
            logvar = t.clip(logvar, -1e5, 1e5)
        else:
            logvar = None

        return dict(mu=mu, logvar=logvar)
    
class EncoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_layers: int,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    n_concepts: int = 0,
    variational: bool = True,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim + n_concepts
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.variational = variational

        self.x_embedder = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [MLPBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)

        if self.variational:
            self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        else:
            self.fc_var = None

    def forward(self, x, concepts=None, **kwargs):

        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h,)

        mu = self.fc_mu(h)
        if self.variational:
            logvar = self.fc_var(h)
            logvar = t.clip(logvar, -1e5, 1e5)
        else:
            logvar = None

        return dict(mu=mu, logvar=logvar)


class CVAEEncoderBlock(nn.Module):
    def __init__(
    self,
    input_dim: int,
    n_layers: int,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    n_concepts: int = 0,
    **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.x_embedder = nn.Linear(self.input_dim + self.n_concepts, hidden_dim)

        self.layers = nn.ModuleList(
            [NoResMLPBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x, input_concept, **kwargs):

        x = torch.concat((x, input_concept), dim=1)

        h = self.x_embedder(x)
        for layer in self.layers:
            h = layer(h,)

        mu, logvar = self.fc_mu(h), self.fc_var(h)
        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)




class DefaultEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
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
        self.latent_dim = latent_dim
        self.dropout = dropout

        layers = []
        layers_dim = [self.input_dim] + self.hidden_dim

        for k in range(0, len(layers_dim) - 1):

            layer_k = [
                nn.Linear(layers_dim[k], layers_dim[k + 1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            layers += layer_k

        self.encoder_layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim[-1], self.latent_dim)

    def forward(self, x, **kwargs):
        h = self.encoder_layers(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)


class ConditionalEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        n_concepts: int = 0,
        **kwargs,
    ):

        super().__init__()

        self.input_dim = input_dim + n_concepts
        self.hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim]
        )
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.dropout = dropout

        layers = []
        layers_dim = [self.input_dim] + self.hidden_dim

        for k in range(0, len(layers_dim) - 1):

            layer_k = [
                nn.Linear(layers_dim[k], layers_dim[k + 1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]
            layers += layer_k

        self.encoder_layers = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim[-1], self.latent_dim)

    def forward(self, x, concepts=None, **kwargs):

        h0 = t.cat((x, concepts), dim=1)
        h = self.encoder_layers(h0)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)
