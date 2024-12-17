import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb

from .base import BaseCBVAE
from .utils import sigmoid
from .encoder import DefaultEncoderBlock
from .decoder import DefaultDecoderBlock, SkipDecoderBlock, SkipDecoderBlock2

EPS = 1e-6


class CB_VAE(BaseCBVAE):
    def __init__(
        self,
        config,
        _encoder: nn.Module = DefaultEncoderBlock,
        _decoder: nn.Module = DefaultDecoderBlock,
        **kwargs,
    ):

        super().__init__(
            config,
            **kwargs,
        )

        self.dropout = config.get("dropout", 0.0)

        # Encoder

        self._encoder = _encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        if "n_unknown" in config:
            n_unknown = config["n_unknown"]
        elif "min_bottleneck_size" in config:
            n_unknown = max(config.min_bottleneck_size, self.n_concepts)
        else:
            n_unknown = 32

        if "cb_layers" in config:
            cb_layers = config["cb_layers"]
        else:
            cb_layers = 1

        cb_concepts_layers = []
        cb_unk_layers = []

        for k in range(0, cb_layers - 1):

            layer_k = [
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]

            cb_concepts_layers += layer_k
            cb_unk_layers += layer_k

        cb_concepts_layers.append(nn.Linear(self.latent_dim, self.n_concepts))
        cb_concepts_layers.append(nn.Sigmoid())

        cb_unk_layers.append(nn.Linear(self.latent_dim, n_unknown))
        cb_unk_layers.append(nn.ReLU())

        self.cb_concepts_layers = nn.Sequential(*cb_concepts_layers)
        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_concepts=self.n_concepts,
            n_unknown=n_unknown,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.dropout = nn.Dropout(p=self.dropout)

        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp

        self.use_orthogonality_loss = config.get("use_orthogonality_loss", True)
        self.use_concept_loss = config.get("use_concept_loss", True)

        if config.get("use_soft_concepts", False):
            self.concept_loss = self._soft_concept_loss
            self.concept_transform = sigmoid
        else:
            self.concept_loss = self._hard_concept_loss
            self.concept_transform = sigmoid

        self.save_hyperparameters()

    @property
    def has_concepts(
        self,
    ):
        return True

    def _extra_loss(self, loss_dict, *args, **kwargs):
        return loss_dict

    def encode(self, x, **kwargs):
        return self._encoder(x, **kwargs)

    def decode(self, h, **kwargs):

        return self._decoder(h, **kwargs)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):
        known_concepts = self.cb_concepts_layers(z)
        unknown = self.cb_unk_layers(z)

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts == None:
                input_concept = known_concepts
            else:
                input_concept = concepts

        h = torch.cat((input_concept, unknown), 1)

        return dict(
            input_concept=input_concept,
            pred_concept=known_concepts,
            unknown=unknown,
            h=h,
        )

    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(**cbm)
        return dec

    def orthogonality_loss(self, c, u):

        batch_size = u.size(0)

        # Compute means along the batch dimension
        u_mean = u.mean(dim=0, keepdim=True)
        c_mean = c.mean(dim=0, keepdim=True)

        # Center the variables in-place to save memory
        u_centered = u - u_mean
        c_centered = c - c_mean

        # Compute the cross-covariance matrix using batch dimension
        cross_covariance = torch.matmul(u_centered.T, c_centered) / (batch_size - 1)

        # Frobenius norm squared of the cross-covariance matrix
        loss = (cross_covariance**2).sum()

        return loss

    def rec_loss(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction="mean")

    def _soft_concept_loss(self, pred_concept, concepts, **kwargs):
        overall_concept_loss = self.n_concepts * F.mse_loss(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

    def _hard_concept_loss(self, pred_concept, concepts, **kwargs):
        overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

    def loss_function(
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        # concept_proj,
        unknown,
        **kwargs,
    ):
        loss_dict = {}

        MSE = self.rec_loss(x_pred, x)
        KLD = self.KL_loss(mu, logvar)

        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        loss_dict["Total_loss"] = MSE + self.beta * KLD

        pred_concept_clipped = t.clip(pred_concept, 0, 1)

        if self.use_concept_loss:
            overall_concept_loss = self.concept_loss(pred_concept_clipped, concepts)
            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:
            orth_loss = self.orthogonality_loss(pred_concept_clipped, unknown)
            loss_dict["orth_loss"] = orth_loss
            loss_dict["Total_loss"] += self.orthogonality_hp * orth_loss

        loss_dict = self._extra_loss(
            loss_dict,
            x=x,
            concepts=concepts,
            x_pred=x_pred,
            mu=mu,
            logvar=logvar,
            pred_concept=pred_concept,
            unknown=unknown,
            **kwargs,
        )

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
    def __init__(self, config, **kwargs):
        super().__init__(config, _decoder=SkipDecoderBlock, **kwargs)

    def decode(self, input_concept, unknown, **kwargs):
        return self._decoder(input_concept, unknown, **kwargs)


class SKIP_SINGLE_CB_VAE(CB_VAE):
    def __init__(self, config):
        super().__init__(config, _decoder=SkipDecoderBlock2)

    def decode(self, input_concept, unknown, **kwargs):
        return self._decoder(input_concept, unknown, **kwargs)


class BACKPACK_CB_VAE(CB_VAE):
    def __init__(self, config):
        super().__init__(config, _decoder=SkipDecoderBlockBackpack)

    def forward(self, x, concepts=None, **kwargs):
        enc = self.encode(x, concepts=concepts, **kwargs)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, concepts=concepts, **enc)
        dec = self.decode(x, **enc, **z, **cbm, concepts=concepts)

        out = dict()
        for d in [enc, z, cbm, dec]:
            out.update(d)
        return out

    def decode(self, x, input_concept, unknown, **kwargs):
        return self._decoder(x, input_concept, unknown, **kwargs)

    def intervene(self, x, concepts, mask, **kwargs):
        enc = self.encode(x)
        z = self.reparametrize(**enc)
        cbm = self.cbm(**z, **enc, concepts=concepts, mask=mask, intervene=True)
        dec = self.decode(x, **cbm)
        return dec


class ALEA_CB_VAE(CB_VAE):
    def __init__(self, config, _decoder=DefaultDecoderBlock):
        super().__init__(config, _decoder=_decoder)

        self.a = nn.Parameter(6 * t.ones(self.n_concepts), requires_grad=True)
        self.cos = nn.CosineSimilarity(dim=0)

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        w = sigmoid(self.a, self.sigmoid_temp)
        self.w = t.clip(w, 1e-3, 1)

        known_concepts = sigmoid(self.fcCB1(z), self.sigmoid_temp)

        known_concepts_proj = self.fcCBproj(known_concepts)
        unknown = F.relu(self.fcCB2(z))

        if intervene:
            concepts_ = known_concepts * (1 - mask) + concepts * mask
            h = torch.cat((w * concepts_, unknown), 1)
        else:
            if concepts == None:
                h = torch.cat((w * known_concepts, unknown), 1)
            else:
                h = torch.cat((w * concepts, unknown), 1)

        return dict(
            pred_concept=known_concepts,
            concept_proj=known_concepts_proj,
            unknown=unknown,
            h=h,
        )

    def concept_loss(self, pred_concept, concepts):
        overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
            pred_concept, concepts, weight=self.w.detach(), reduction="mean"
        )
        return overall_concept_loss

    def _extra_loss(self, loss_dict, pred_concept, **kwargs):

        log_B = t.log(t.tensor(pred_concept.size(0)))

        s = pred_concept / t.sum(pred_concept, dim=0)
        e = -t.sum(t.log(s + 1e-8) * s, dim=0)
        e = (log_B - e) / log_B

        alea_loss = 1 - self.cos(self.w, e)

        loss_dict["Alea_loss"] = alea_loss
        loss_dict["Total_loss"] += alea_loss

        return loss_dict


class SKIP_ALEA_CB_VAE(ALEA_CB_VAE):
    def __init__(self, config):
        super().__init__(config, _decoder=SkipDecoderBlock)
