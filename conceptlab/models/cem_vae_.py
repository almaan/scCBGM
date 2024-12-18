import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t


# Define the VAE model
class CEM_VAE(pl.LightningModule):
    def __init__(self, config):
        super(CEM_VAE, self).__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.n_concepts = config.n_concepts
        self.learning_rate = config.lr
        self.independent_training = config.independent_training
        self.emb_size = config.emb_size
        self.concept_bins = config.concept_bins

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # CBM
        self.sigmoid = torch.nn.Sigmoid()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.concept_context_generators = torch.nn.ModuleList()

        for c in range(self.n_concepts):

            self.concept_context_generators.append(
                torch.nn.Sequential(
                    *[
                        torch.nn.Linear(
                            self.latent_dim, self.concept_bins[c] * self.emb_size
                        ),
                        nn.BatchNorm1d(self.concept_bins[c] * self.emb_size),
                    ]
                )
            )

            self.concept_prob_generators.append(
                torch.nn.Sequential(
                    *[
                        torch.nn.Linear(
                            self.concept_bins[c] * self.emb_size, self.concept_bins[c]
                        )
                    ]
                )
            )

        self.concept_context_generators.append(
            torch.nn.Sequential(
                *[
                    torch.nn.Linear(self.latent_dim, self.emb_size),
                    nn.BatchNorm1d(self.emb_size),
                ]
            )
        )

        self.g_latent = self.emb_size * (self.n_concepts + 1)
        # self.g_latent+=sum(self.concept_bins)

        # self.fcCB1 = nn.Linear(self.latent_dim,  self.n_concepts)
        # self.fcCB2 = nn.Linear(self.latent_dim,  self.n_concepts)
        # Decoder
        self.fc3 = nn.Linear(self.g_latent, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = config.beta
        self.concepts_hp = config.concepts_hp
        self.orthogonality_hp = config.orthogonality_hp

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

    def cbm(self, h, concepts=None, mask=None, intervene=False):

        non_concept_latent = None
        all_concept_latent = None
        all_concepts = None
        all_logits = None

        for c in range(self.n_concepts + 1):
            ### 1 generate context
            context = self.concept_context_generators[c](h)

            if c < self.n_concepts:
                ### 2 get prob given concept
                if concepts == None:
                    logits = self.concept_prob_generators[c](context)
                    prob_gumbel = self.sigmoid(logits)
                else:
                    if intervene:
                        logits = self.concept_prob_generators[c](context)
                        prob_gumbel = self.sigmoid(logits)
                        mask_c = mask[:, c].unsqueeze(-1)
                        concepts_c = concepts[:, c].unsqueeze(-1)

                        prob_gumbel = prob_gumbel * (1 - mask_c) + concepts_c * mask_c

                    else:
                        logits = concepts[:, c].unsqueeze(-1)
                        prob_gumbel = concepts[:, c].unsqueeze(-1)
                for i in range(self.concept_bins[c]):
                    temp_concept_latent = context[
                        :, (i * self.emb_size) : ((i + 1) * self.emb_size)
                    ] * prob_gumbel[:, i].unsqueeze(-1)
                    if i == 0:
                        concept_latent = temp_concept_latent
                    else:
                        concept_latent = concept_latent + temp_concept_latent

                if all_concept_latent == None:
                    all_concept_latent = concept_latent
                else:
                    all_concept_latent = torch.cat(
                        (all_concept_latent, concept_latent), 1
                    )

                if all_concepts == None:
                    all_concepts = prob_gumbel
                    all_logits = logits
                else:
                    all_concepts = torch.cat((all_concepts, prob_gumbel), 1)
                    all_logits = torch.cat((all_logits, logits), 1)

            else:
                for c in range(self.n_concepts):
                    if c == 0:
                        non_concept_latent = context
                    else:
                        non_concept_latent = torch.cat((non_concept_latent, context), 1)

        z = torch.cat((all_concept_latent, context), 1)
        return all_concept_latent, non_concept_latent, all_concepts, z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x, concepts=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        known_concepts, unknown, pred_concept, z = self.cbm(z, concepts)
        x_pred = self.decode(z)
        return {
            "x_pred": x_pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "pred_concept": pred_concept,
            "known_concepts": known_concepts,
            "unknown": unknown,
        }

    def intervene(self, x, concepts, mask):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        known_concepts, unknown, pred_concept, z = self.cbm(z, concepts, mask, True)
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
        self,
        x,
        concepts,
        x_pred,
        mu,
        logvar,
        pred_concept,
        known_concepts,
        unknown,
        **kwargs,
    ):
        loss_dict = {}
        MSE = F.mse_loss(x_pred, x, reduction="mean")

        overall_concept_loss = self.n_concepts * F.mse_loss(
            pred_concept, concepts, reduction="mean"
        )

        for c in range(self.n_concepts):
            accuracy = self.binary_accuracy(pred_concept[:, c], concepts[:, c])
            loss_dict[str(c) + "_acc"] = accuracy

        orth_loss = self.orthogonality_loss(known_concepts, unknown)
        loss_dict["concept_loss"] = overall_concept_loss
        loss_dict["orth_loss"] = orth_loss
        KLD = self.KL_loss(mu, logvar)
        loss_dict["rec_loss"] = MSE
        loss_dict["KL_loss"] = KLD
        loss_dict["Total_loss"] = (
            MSE
            + self.beta * KLD
            + self.concepts_hp * overall_concept_loss
            + self.orthogonality_hp * orth_loss
        )

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
