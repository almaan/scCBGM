import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR

from abc import ABC, abstractmethod


class BaseCBVAE(pl.LightningModule, ABC):
    def __init__(self, config):

        super(BaseCBVAE, self).__init__()

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.n_concepts = config.n_concepts
        self.learning_rate = config.lr
        self.independent_training = config.get("independent_training", False)
        self.beta = config.beta

    @property
    @abstractmethod
    def has_concepts(self):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def cbm(self, *args, **kwargs):
        pass

    @abstractmethod
    def intervene(self, x, concepts, mask):
        pass

    def log_parameters(self, *args, **kwargs):
        return None

    def reparametrize(self, mu, logvar, **kwargs):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return dict(z=mu + eps * std)

    def forward(self, x, concepts=None, **kwargs):
        enc = self.encode(x, concepts=concepts, **kwargs)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, concepts=concepts, **enc)
        dec = self.decode(**enc, **z, **cbm, concepts=concepts)

        out = dict()
        for d in [enc, z, cbm, dec]:
            out.update(d)

        return out

    def _step(self, batch, batch_idx, prefix="train"):

        x, concepts = batch
        if prefix == "train" and self.independent_training:
            out = self(x, concepts)
        else:
            out = self(x, concepts=concepts)

        loss_dict = self.loss_function(x, concepts, **out)

        for key, value in loss_dict.items():
            self.log(f"{prefix}_{key}", value)

        return loss_dict["Total_loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define the CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    def binary_accuracy(self, preds, labels, threshold=0.5):
        """
        Calculate the binary accuracy of a batch.
        Args:
        preds (torch.Tensor): The predicted probabilities or logits.
        labels (torch.Tensor): The true labels.
        threshold (float): The threshold to convert probabilities to binary predictions.
        Returns:
        float: The binary accuracy.
        """

        # Convert probabilities to binary predictions
        binary_preds = (preds >= threshold).float()

        # Calculate the number of correct predictions
        correct = (binary_preds == labels).float().sum()

        # Calculate the accuracy
        accuracy = correct / labels.numel()

        return accuracy.item()
