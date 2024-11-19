import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base import BaseCBVAE
import wandb


class SCBM(BaseCBVAE):
    def __init__(self, config):
        super().__init__(config)

        self.amortized = config.get("amortized", True)

        #  -- Encoder --
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # -- CB --
        self.n_unknown = config.get("n_unknown", self.n_concepts)

        # known concepts
        self.fcCBmu = nn.Linear(self.latent_dim, self.n_concepts)

        # setup for amortized or global covariance
        if self.amortized:
            self.fcCBcov = nn.Linear(
                self.latent_dim, ((self.n_concepts) * (self.n_concepts + 1)) // 2
            )
            self._cbm = self._amortized_cbm
        else:
            self.L_flat = nn.Parameter(
                t.randn(((self.n_concepts) * (self.n_concepts + 1)) // 2),
                requires_grad=True,
            )
            self.L_flat = self.L_flat.to(self.device)
            self._cbm = self._global_cbm

        self.cov_tril_indices = t.tril_indices(self.n_concepts, self.n_concepts)

        self.fcCBproj = nn.Linear(self.n_concepts, self.n_unknown)

        # unknown concepts
        self.fcCB2 = nn.Linear(self.latent_dim, self.n_unknown)

        #  -- Decoder --
        layers = []
        in_dim = self.n_concepts + self.n_unknown

        for i in range(self.n_decoder_layers):
            out_dim = (
                self.input_dim if i == self.n_decoder_layers - 1 else self.hidden_dim
            )
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = self.hidden_dim

        self.decoder_layers = nn.ModuleList(layers)

        # -- hyperparams --
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp
        self.dropout = config.get("dropout", 0.0)
        self.precision_loss = 100

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        self.save_hyperparameters()

    @property
    def has_concepts(
        self,
    ):
        return True

    def encode(self, x, **kwargs):
        h = self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=True, inplace=False)

        mu, logvar = self.fc21(h), self.fc22(h)

        logvar = t.clip(logvar, -1e5, 1e5)

        return dict(mu=mu, logvar=logvar)

    def _cbm_tail(self, z, cb_mu, eta, concepts, mask, intervene, **kwargs):

        if self.train:
            U = t.rand((cb_mu.size(0), self.n_concepts), device=self.device)
            G = -t.log(-t.log(U))
            known_concepts = F.sigmoid(eta + G)
        else:
            known_concepts = F.sigmoid(eta)

        known_concepts_proj = self.fcCBproj(known_concepts)

        unknown = self.fcCB2(z)
        unknown = F.relu(unknown)

        if self.scale_concept:
            known_concepts_bar = (
                known_concepts * self.learned_concept_scale_gamma
                + self.learned_concept_scale_beta
            )
            if concepts is not None:
                concepts_bar = (
                    concepts * self.learned_concept_scale_gamma
                    + self.learned_concept_scale_beta
                )

        else:
            known_concepts_bar = known_concepts
            concepts_bar = concepts

        if intervene:
            concepts_ = known_concepts_bar * (1 - mask) + concepts_bar * mask
            h = torch.cat((concepts_, unknown), 1)
        else:
            if concepts == None:
                h = torch.cat((known_concepts_bar, unknown), 1)
            else:
                h = torch.cat((concepts_bar, unknown), 1)

        return dict(
            pred_concept=known_concepts,
            concept_proj=known_concepts_proj,
            unknown=unknown,
            h=h,
        )

    def _global_cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        cb_mu = self.fcCBmu(z)

        L = t.zeros((self.n_concepts, self.n_concepts), device=self.device)
        L[self.cov_tril_indices[0], self.cov_tril_indices[1]] = self.L_flat

        eps = torch.randn_like(cb_mu)
        eta = cb_mu + eps @ L.T

        return self._cbm_tail(z, cb_mu, eta, concepts, mask, intervene)

    def _amortized_cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        cb_mu = self.fcCBmu(z)
        cb_cov = self.fcCBcov(z)

        L = t.zeros((z.size(0), self.n_concepts, self.n_concepts), device=self.device)
        L[:, self.cov_tril_indices[0], self.cov_tril_indices[1]] = t.flatten(
            cb_cov, start_dim=1
        )

        eps = torch.randn_like(cb_mu)
        eta = cb_mu + t.einsum("nab,nb->nb", L, eps)

        return self._cbm_tail(z, cb_mu, eta, concepts, mask, intervene)

    def cbm(self, *args, **kwargs):
        return self._cbm(*args, **kwargs)

    def decode(self, h, **kwargs):
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i < self.n_decoder_layers - 1:  # Apply dropout to all but the last layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

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

        MSE = F.mse_loss(x_pred, x, reduction="mean")

        KLD = self.KL_loss(mu, logvar)
        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        loss_dict["Total_loss"] = MSE + self.beta * KLD

        if self.use_concept_loss:

            overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
                pred_concept, concepts, reduction="mean"
            )

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
