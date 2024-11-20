import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR
from wandb import Histogram

from .base import BaseCBVAE
from .utils import sigmoid
from .encoder import DefaultEncoderBlock
from .decoder import DefaultDecoderBlock, SkipDecoderBlock


class CB_VAE(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = DefaultEncoderBlock,
        _decoder: nn.Module = DefaultDecoderBlock,
    ):

        super().__init__(
            config,
        )

        self.dropout = config.get("dropout", 0.0)

        # Encoder

        self._encoder = _encoder(
            input_dim=self.input_dim,
            n_concepts=self.n_concepts,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        if "n_unknown" in config:
            n_unknown = config["n_unknown"]
        elif "min_bottleneck_size" in config:
            n_unknown = max(
                config.min_bottleneck_size - self.n_concepts, self.n_concepts
            )
        else:
            n_unknown = 32

        self.fcCB1 = nn.Linear(self.latent_dim, self.n_concepts)
        self.fcCBproj = nn.Linear(self.n_concepts, n_unknown)
        self.fcCB2 = nn.Linear(self.latent_dim, n_unknown)

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_concepts=self.n_concepts,
            n_unknown=n_unknown,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        self.save_hyperparameters()

    @property
    def has_concepts(
        self,
    ):
        return True

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)

    def decode(self, h, **kwargs):
        return self._decoder(h, **kwargs)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        known_concepts = sigmoid(self.fcCB1(z), self.sigmoid_temp)
        known_concepts_proj = self.fcCBproj(known_concepts)
        unknown = F.relu(self.fcCB2(z))

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

    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
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

        pred_concept_clipped = t.clip(pred_concept, 0, 1)

        if self.use_concept_loss:

            overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
                pred_concept_clipped, concepts, reduction="mean"
            )

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

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
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


class SKIP_CB_VAE(CB_VAE):
    def __init__(self, config):
        super().__init__(config, _decoder=SkipDecoderBlock)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        known_concepts = sigmoid(self.fcCB1(z), self.sigmoid_temp)
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
