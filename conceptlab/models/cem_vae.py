import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import torch.nn.utils.parametrizations as param
from wandb import Histogram
import wandb

from .base import BaseCBVAE
from .utils import sigmoid
from .encoder import DefaultEncoderBlock
from .decoder import DefaultDecoderBlock, SkipDecoderBlock


class CEM_VAE(BaseCBVAE):
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

        if self.n_concepts > n_unknown:
            # Case 1: If n_concepts is larger than n_unknown
            n_unknown = self.n_concepts
            self.emb_size = 1
        else:
            # Case 2: Make n_concept * emb_size = n_unknown
            # Since we can't modify n_concepts, we'll adjust emb_size
            self.emb_size = n_unknown // self.n_concepts
            # Adjust n_unknown to be exactly divisible by n_concepts
            n_unknown = self.n_concepts * self.emb_size

        # Create separate context generators for each concept
        self.concept_context_generators = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.latent_dim, 2 * self.emb_size), nn.ReLU())
                for _ in range(self.n_concepts)
            ]
        )

        # Separate probability generator for each concept
        self.concept_prob_generators = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * self.emb_size, 1), nn.Sigmoid())
                for _ in range(self.n_concepts)
            ]
        )

        if "cb_layers" in config:
            cb_layers = config["cb_layers"]
        else:
            cb_layers = 1

        cb_unk_layers = []

        for k in range(0, cb_layers - 1):

            layer_k = [
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
            ]

            cb_unk_layers += layer_k

        cb_unk_layers.append(nn.Linear(self.latent_dim, n_unknown))
        cb_unk_layers.append(nn.ReLU())

        self.cb_unk_layers = nn.Sequential(*cb_unk_layers)

        self._decoder = _decoder(
            input_dim=self.input_dim,
            n_concepts=n_unknown,
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

        self.use_concept_loss = config.get("use_concept_loss", True)

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

    def concept_loss(self, pred_concept, concepts, **kwargs):

        overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
            pred_concept, concepts, reduction="mean"
        )
        return overall_concept_loss

    def cbm(self, z, concepts=None, mask=None, intervene=False, **kwargs):

        unknown = self.cb_unk_layers(z)

        # Lists to store results
        contexts = []
        probs = []

        # Generate context and probability for each concept
        for i in range(self.n_concepts):
            # Generate context
            context = self.concept_context_generators[i](
                z
            )  # shape: [..., 2 * emb_size]
            contexts.append(context)

            # Generate probability from this concept's context
            prob = self.concept_prob_generators[i](context)  # shape: [..., 1]
            probs.append(prob)

        # Stack results
        contexts = torch.stack(
            contexts, dim=-2
        )  # shape: [..., n_concepts, 2 * emb_size]
        contexts = contexts.unsqueeze(-2) if contexts.ndimension() == 2 else contexts

        known_concepts = torch.stack(probs, dim=-2).squeeze(
            -1
        )  # Ensure shape [..., n_concepts]
        known_concepts = (
            known_concepts.unsqueeze(-1)
            if known_concepts.ndimension() == 1
            else known_concepts
        )

        pos_context = context[..., : self.emb_size]  # shape: [..., emb_size]
        neg_context = context[..., self.emb_size :]  # shape: [..., emb_size]

        # Expand contexts to match number of concepts
        pos_context = pos_context.unsqueeze(-2).expand(
            *pos_context.shape[:-1], self.n_concepts, self.emb_size
        )
        neg_context = neg_context.unsqueeze(-2).expand(
            *neg_context.shape[:-1], self.n_concepts, self.emb_size
        )

        if intervene:
            input_concept = known_concepts * (1 - mask) + concepts * mask
        else:
            if concepts == None:
                input_concept = known_concepts
            else:
                input_concept = concepts

        input_concept = input_concept.unsqueeze(-1).expand(
            *input_concept.shape, self.emb_size
        )
        # Weight contexts with probabilities
        weighted_pos = pos_context * input_concept
        weighted_neg = neg_context * (1 - input_concept)

        # Combine weighted contexts
        combined = weighted_pos + weighted_neg  # shape: [..., n_concepts, emb_size]
        # Reshape to have size emb_size * n_concepts

        final_shape = list(combined.shape[:-2]) + [self.emb_size * self.n_concepts]

        emd_concept = combined.reshape(*final_shape)
        h = torch.cat((emd_concept, unknown), 1)

        return dict(
            pred_concept=known_concepts,
            emd_concept=emd_concept,
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
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(c, u))
        return output.mean()

    def rec_loss(self, x_pred, x):
        return F.mse_loss(x_pred, x, reduction="mean")

    def loss_function(
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        emd_concept,
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
            orth_loss = self.orthogonality_loss(emd_concept, unknown)
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
        scheduler = ExponentialLR(optimizer, gamma=0.997)

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
