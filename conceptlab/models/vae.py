import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR
from .base import BaseCBVAE
from .encoder import DefaultEncoderBlock
from .decoder import DefaultDecoderBlock


class VAE(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = DefaultEncoderBlock,
        _decoder: nn.Module = DefaultDecoderBlock,
    ):

        super().__init__(
            config,
        )

        self.beta = config.beta
        self.dropout = config.get("dropout", 0.0)

        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_unknown=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        self.save_hyperparameters()

    @property
    def has_concepts(
        self,
    ):
        return False

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)

    def cbm(self, z, **kwargs):
        return dict(h=z)

    def intervene(self, x, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z)
        dec = self.decode(**cbm)
        return dec

    def decode(self, h, **kwargs):
        return self._decoder(h, **kwargs)

    def loss_function(self, x, concepts, x_pred, mu, logvar, **kwargs):

        loss_dict = {}
        MSE = F.mse_loss(x_pred, x, reduction="mean")
        KLD = self.KL_loss(mu, logvar)
        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD
        loss_dict["Total_loss"] = MSE + self.beta * KLD

        return loss_dict

    def configure_optimizers(
        self,
    ):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define the CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        # Return a dictionary with the optimizer and the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # The LR scheduler instance
                "interval": "epoch",  # The interval to step the scheduler ('epoch' or 'step')
                "frequency": 1,  # How often to update the scheduler
                "monitor": "val_loss",  # Optional: if you use reduce-on-plateau schedulers
            },
        }
