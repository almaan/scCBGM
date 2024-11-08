import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import BaseCBVAE


class CVAE(BaseCBVAE):
    def __init__(self, config):
        super().__init__(
            config,
        )
        # Encoder
        self.fc1 = nn.Linear(self.input_dim + self.n_concepts, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(self.latent_dim + self.n_concepts, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = config.beta
        self.dropout = config.get("dropout", 0.3)

        self.save_hyperparameters()

    @property
    def has_concepts(
        self,
    ):
        return True

    def encode(self, x, concepts, **kwargs):

        h0 = t.cat((x, concepts), dim=1)
        h1 = F.relu(self.fc1(h0))
        h1 = F.dropout(h1, p=self.dropout, training=True, inplace=False)
        mu, logvar = self.fc21(h1), self.fc22(h1)
        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)

    def cbm(self, z, **kwargs):
        return dict(h=z)

    def intervene(self, x, concepts, mask, **kwargs):
        _concepts = concepts * (1 - mask) + concepts * mask
        enc = self.encode(x, concepts=concepts)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm, concepts=_concepts)
        return dec

    def decode(self, h, concepts, **kwargs):

        h0 = t.cat((h, concepts), dim=1)

        h3 = F.relu(self.fc3(h0))
        h3 = F.dropout(h3, p=self.dropout, training=True, inplace=False)
        h4 = self.fc4(h3)
        return dict(x_pred=h4)

    def loss_function(self, x, concepts, x_pred, mu, logvar, **kwargs):

        loss_dict = {}
        MSE = F.mse_loss(x_pred, x, reduction="mean")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
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
