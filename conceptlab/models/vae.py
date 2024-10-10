import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR



# Define the VAE model
class VAE(pl.LightningModule):
    def __init__(self, config):
        super(VAE, self).__init__()

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.learning_rate = config.lr

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = config.beta
        self.dropout = config.get("dropout", 0.3)
        self.save_hyperparameters()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        logvar = t.clip(logvar, -1e5, 1e5)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        return {"x_pred": x_pred, "z": z, "mu": mu, "logvar": logvar}

    def loss_function(self, x, x_pred, mu, logvar, **kwargs):
        loss_dict = {}
        MSE = F.mse_loss(x_pred, x, reduction="mean")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD
        loss_dict["Total_loss"] = MSE + self.beta * KLD

        return loss_dict

    def _step(self, batch, batch_idx, prefix="train"):
        x = batch
        out = self(x)
        loss_dict = self.loss_function(x, **out)

        for key, value in loss_dict.items():
            self.log(f"{prefix}_{key}", value)
        return loss_dict["Total_loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

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
