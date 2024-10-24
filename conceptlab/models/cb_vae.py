import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import BaseCBVAE


class CB_VAE(BaseCBVAE):
    def __init__(self, config):

        super().__init__(config)

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        self.fcCB1 = nn.Linear(self.latent_dim, self.n_concepts)

        if "n_unknown" in config:
            n_unknown = config["n_unknown"]
        elif "min_bottleneck_size" in config:
            n_unknown = max(
                config.min_bottleneck_size - self.n_concepts, self.n_concepts
            )
        else:
            n_unknown = 32

        self.fcCBproj = nn.Linear(self.n_concepts, n_unknown)
        self.fcCB2 = nn.Linear(self.latent_dim, n_unknown)

        # Decoder
        self.fc3 = nn.Linear((self.n_concepts + n_unknown), self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp
        self.dropout = config.get("dropout", 0.0)

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        self.save_hyperparameters()

    def encode(self, x, **kwargs):
        h1 = F.relu(self.fc1(x))
        h1 = F.dropout(h1, p=self.dropout, training=True, inplace=False)
        mu, logvar = self.fc21(h1), self.fc22(h1)
        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        known_concepts = F.sigmoid(self.fcCB1(z))
        known_concepts_proj = self.fcCBproj(known_concepts)
        unknown = F.relu(self.fcCB2(z))

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
        h3 = F.relu(self.fc3(h))
        h3 = F.dropout(h3, p=self.dropout, training=True, inplace=False)
        h4 = self.fc4(h3)
        return dict(x_pred=h4)

    def intervene(self, x, concepts, mask, **kwargs):
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
        MSE = F.mse_loss(x_pred, x, reduction="mean")

        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

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
