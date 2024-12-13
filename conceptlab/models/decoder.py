import torch.nn as nn
import torch.nn.functional as F
import torch as t


class DefaultDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_unknown: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        n_concepts: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim


        self.n_layers=n_layers

        self.hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim for i in range (self.n_layers)]
        )
        self.n_concepts = n_concepts
        self.n_unknown = n_unknown
        self.dropout = dropout

        layers = []
        layers_dim = [self.n_concepts + self.n_unknown] + self.hidden_dim

        for k in range(0, self.n_layers - 1):
            if k ==0:
                in_dim=self.n_concepts + self.n_unknown
            else:
                in_dim=self.hidden_dim[k-1]

            layer_k = [
                nn.Linear(in_dim, self.hidden_dim[k]),
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
    def __init__(
        self,
        input_dim, 
        hidden_dim,
        dropout_p=0.0,
        is_last_layer=False
    ):
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
        n_layers: int,
        dropout: float = 0.0,
        n_concepts: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.n_layers=n_layers
        self.n_concepts = n_concepts

        self.hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (list, tuple)) else [hidden_dim for i in range (self.n_layers)]
        )

        self.n_unknown = n_unknown
        self.dropout = dropout

        layers = []

        for k in range(0, self.n_layers - 1):
            if k ==0:
                in_dim= self.n_unknown
            else:
                in_dim=self.hidden_dim[k-1]

            layers.append(
                SkipLayer(å
                    layers_dim[k],
                    layers_dim[k + 1],
                    self.dropout,
                    is_last_layer=False,
                )
            )

        layers.append(
            SkipLayer(
                self.hidden_dim[-1]+self.n_concepts,
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
