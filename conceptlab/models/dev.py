import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import BaseCBVAE


# Define the VAE model
class DEV(BaseCBVAE):
    def __init__(self, config):
        super().__init__(config)

        #  -- Encoder --
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # -- CB --
        self.n_unknown = config.get("n_unknown", self.n_concepts)

        # known concepts
        self.fcCB1 = nn.Linear(self.latent_dim, self.n_concepts)
        self.fcCBproj = nn.Linear(self.n_concepts, self.n_unknown)

        # unknown concepts
        self.fcCB2 = nn.Linear(self.latent_dim, self.n_unknown)

        #  -- Decoder --
        self.fc3 = nn.Linear((self.n_concepts + self.n_unknown), self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

        # -- hyperparams --
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp
        self.dropout = config.get("dropout", 0.0)

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        self.save_hyperparameters()

    def encode(self, x, **kwargs):
        h = self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=True, inplace=False)

        mu, logvar = self.fc21(h), self.fc22(h)

        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        known_concepts = self.fcCB1(z)
        known_concepts = F.sigmoid(known_concepts)
        known_concepts_proj = self.fcCBproj(known_concepts)

        unknown = self.fcCB2(z)
        unknown = F.relu(unknown)

        if intervene:
            concepts_ = known_concepts * (1 - mask) + concepts * mask
            h = torch.cat((concepts_, unknown), 1)
        else:
            if concepts == None:
                h = torch.cat((known_concepts, unknown), 1)
            else:
                h = torch.cat((concepts, unknown), 1)

        return dict(
            pred_concept=known_concepts,
            concept_proj=known_concepts_proj,
            unknown=unknown,
            h=h,
        )

    def decode(self, h, **kwargs):
        h = self.fc3(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=True, inplace=False)
        h = self.fc4(h)
        return dict(x_pred=h)

    def intervene(self, x, concepts, mask):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm)

        return dec

    def orthogonality_loss(self, concept_emb, unk_emb):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(concept_emb, unk_emb))
        return output.mean()

    def loss_function(
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        concept_proj,
        unknown,
        **kwargs,
    ):
        loss_dict = {}
        self.log("fcCB2", float(self.fcCB2.weight[0, 0]))

        MSE = F.mse_loss(x_pred, x, reduction="mean")

        KLD = self.KL_loss(mu, logvar)

        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        loss_dict["Total_loss"] = MSE + self.beta * KLD

        if self.use_concept_loss:

            overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
                pred_concept, concepts, reduction="mean"
            )

            for c in range(self.n_concepts):
                accuracy = self.binary_accuracy(pred_concept[:, c], concepts[:, c])
                loss_dict[str(c) + "_acc"] = accuracy

            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:

            orth_loss = self.orthogonality_loss(concept_proj, unknown)
            loss_dict["orth_loss"] = orth_loss
            loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss

        return loss_dict

    def _step(self, batch, batch_idx, prefix="train"):
        x, concepts = batch
        if prefix == "train" and self.independent_training:
            out = self(x, concepts)

        else:
            out = self(x)
        loss_dict = self.loss_function(x, concepts, **out)

        for key, value in loss_dict.items():
            self.log(f"{prefix}_{key}", value)
        return loss_dict["Total_loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def get_params_excluding_layers(self, exclude_layers):
        # exclude_layers is a list of layer names to exclude
        return [
            param
            for name, param in self.named_parameters()
            if name.split(".")[0] not in exclude_layers
        ]

    def configure_optimizers(
        self,
    ):

        standard_params = self.get_params_excluding_layers(["fcCB2"])
        slow_params = [*self.fcCB2.parameters()]

        optimizer = torch.optim.Adam(
            [
                {"params": standard_params, "lr": self.learning_rate},
                {"params": slow_params, "lr": self.learning_rate},
            ]
        )

        # Define the CosineAnnealingLR scheduler
        scheduler1 = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        scheduler2 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1, total_iters=20
        )

        scheduler_config = [
            {
                "scheduler": scheduler1,
                "interval": "epoch",
                "name": "CosineAnnealingLR",
                "param_group": 0,
            },
            {
                "scheduler": scheduler2,
                "interval": "epoch",
                "name": "StepLR",
                "param_group": 1,
            },
        ]

        return [optimizer], scheduler_config
