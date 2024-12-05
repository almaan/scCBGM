import torch.nn as nn
import torch.nn.functional as F
import torch as t


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


class SkipDecoderBlock(nn.Module):
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

        cbm_dim = self.n_concepts + self.n_unknown

        layers = []
        layers_dim = [cbm_dim] + self.hidden_dim

        for k in range(0, len(layers_dim) - 1):

            layer_k = [
                nn.Linear(layers_dim[k], layers_dim[k + 1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]

            layers += layer_k

        self.decoder_layers = nn.Sequential(*layers)

        self.fc_skip = nn.Linear(self.hidden_dim[-1] + cbm_dim, self.input_dim)

    def forward(self, h, **kwargs):

        h_tail = self.decoder_layers(h)
        h_joint = t.cat((h_tail, h), dim=1)
        h_head = self.fc_skip(h_joint)

        return dict(x_pred=h_head)
