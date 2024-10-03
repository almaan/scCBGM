import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torchmetrics.regression import LogCoshError


# Define the VAE model
class CB_VAE(pl.LightningModule):
    def __init__(self, config):
        super(CB_VAE, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.n_concepts = config.n_concepts
        self.learning_rate = config.lr
        self.independent_training = config.independent_training

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        self.fcCB1 = nn.Linear(self.latent_dim, self.n_concepts)

        n_unknown = max(config.min_bottleneck_size - self.n_concepts, self.n_concepts)
        self.fcCB2 = nn.Linear(self.latent_dim, n_unknown)

        # Decoder
        self.fc3 = nn.Linear((self.n_concepts + n_unknown), self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp
        self.dropout = config.get('dropout',0.3)


        self.use_orthogonality_loss = config.get("use_orthogonality_loss", False)
        self.use_concept_loss = config.get("use_concept_loss", True)

        self.save_hyperparameters()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.dropout(h1, p=self.dropout, training=True, inplace=False)
        mu, logvar = self.fc21(h1), self.fc22(h1)
        logvar = t.clip(logvar, -1e5, 1e5)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def cbm(self, z, concepts=None, mask=None, intervene=False):

        known_concepts = F.sigmoid(self.fcCB1(z))
        unknown = F.relu(self.fcCB2(z))

        if intervene:
            concepts_ = known_concepts * (1 - mask) + concepts * mask
            h = torch.cat((concepts_, unknown), 1)
        else:
            if concepts == None:
                h = torch.cat((known_concepts, unknown), 1)
            else:
                h = torch.cat((concepts, unknown), 1)

        return known_concepts, unknown, h

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.dropout(h3, p = self.dropout, training=True, inplace=False)
        return self.fc4(h3)

    def forward(self, x, concepts=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        known_concepts, unknown, z = self.cbm(z, concepts)
        x_pred = self.decode(z)
        return {
            "x_pred": x_pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "pred_concept": known_concepts,
            "unknown": unknown,
        }

    def intervene(self, x, concepts, mask):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        known_concepts, unknown, z = self.cbm(z, concepts, mask, True)
        x_pred = self.decode(z)
        return {"x_pred": x_pred}

    def orthogonality_loss(self, concept_emb, unk_emb):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(concept_emb, unk_emb))
        return output.mean()

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

    def loss_function(
        self, x, concepts, x_pred, mu, logvar, pred_concept, unknown, **kwargs
    ):
        loss_dict = {}
        MSE = F.mse_loss(x_pred, x, reduction="mean")
        # MSE = self.logcosh_loss(x_pred, x).mean()
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD

        loss_dict["Total_loss"] = MSE + self.beta * KLD

        if self.use_concept_loss:

            overall_concept_loss = self.n_concepts * F.binary_cross_entropy(
                pred_concept, concepts, reduction="mean"
            )

            concept_losses = []
            for c in range(self.n_concepts):
                accuracy = self.binary_accuracy(pred_concept[:, c], concepts[:, c])
                loss_dict[str(c) + "_acc"] = accuracy

            loss_dict["concept_loss"] = overall_concept_loss
            loss_dict["Total_loss"] += self.concepts_hp * overall_concept_loss

        if self.use_orthogonality_loss:
            orth_loss = self.orthogonality_loss(pred_concept, unknown)
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

    # def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self,):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define the CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        # Return a dictionary with the optimizer and the scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance
                'interval': 'epoch',  # The interval to step the scheduler ('epoch' or 'step')
                'frequency': 1,       # How often to update the scheduler
                'monitor': 'val_loss', # Optional: if you use reduce-on-plateau schedulers
            }
        }
