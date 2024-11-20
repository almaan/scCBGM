import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch as t


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
