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
