import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch as t


# Define the VAE model
class CB_VAE(pl.LightningModule):
    def __init__(self, input_dim=34455, hidden_dim=1024, latent_dim=512, n_concepts=8, learning_rate=1e-3, beta: float = 1,
                concepts_hp: float = 0.01,
                orthogonality_hp: float = 0.1,
        ):
        super(CB_VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_concepts=n_concepts
        self.learning_rate = learning_rate

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance

        self.fcCB1 = nn.Linear(latent_dim,  self.n_concepts)
        self.fcCB2 = nn.Linear(latent_dim,  self.n_concepts)
        
        # Decoder
        self.fc3 = nn.Linear(2*self.n_concepts, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.beta = beta
        self.concepts_hp =concepts_hp
        self.orthogonality_hp =orthogonality_hp

        self.save_hyperparameters()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu,logvar = self.fc21(h1), self.fc22(h1)
        logvar = t.clip(logvar,-1e5,1e5)
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def cbm (self, z,concepts=None):

        known_concepts=F.relu(self.fcCB1(z))
        unknown=F.relu(self.fcCB2(z))

        if(concepts==None):
            h = torch.cat((known_concepts,unknown),1)
        else:
            h = torch.cat((concepts,unknown),1)


        return known_concepts,unknown, h
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x,concepts=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        known_concepts,unknown, z =self.cbm(z,concepts)
        x_pred = self.decode(z)
        return {'x_pred':x_pred, 'z':z, 'mu':mu, 'logvar':logvar , 'pred_concept':known_concepts, 'unknown':unknown}

    def orthogonality_loss(self, concept_emb,unk_emb):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        output = torch.abs(cos(concept_emb, unk_emb))
        return output.mean()
    def binary_accuracy(self,preds, labels, threshold=0.5):
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
    def loss_function(self, x,concepts, x_pred, mu, logvar,pred_concept,unknown ,**kwargs):
        loss_dict ={}
        MSE = F.mse_loss(x_pred, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        overall_concept_loss =self.n_concepts*F.mse_loss(pred_concept, concepts, reduction="mean")




        concept_losses =[]
        for c in range(self.n_concepts):
            accuracy = self.binary_accuracy(pred_concept[:,c], concepts[:,c])
            loss_dict[str(c)+"_acc"]= accuracy



        orth_loss = self.orthogonality_loss(pred_concept, unknown)
        loss_dict['concept_loss'] = overall_concept_loss
        loss_dict['orth_loss'] = orth_loss

        loss_dict['rec_loss']= MSE
        loss_dict['KL_loss']= KLD
        loss_dict['Total_loss']= MSE + self.beta * KLD + self.concepts_hp*overall_concept_loss+ self.orthogonality_hp*orth_loss
        

        return loss_dict



    def _step(self,batch,batch_idx, prefix = 'train'):
        x,concepts = batch
        if prefix == "train":
            out = self(x,concepts)

        else:
        	out = self(x)
        loss_dict = self.loss_function(x,concepts,**out)

        for key, value in loss_dict.items():
            self.log(f'{prefix}_{key}', value)
        return loss_dict['Total_loss']

    def training_step(self, batch, batch_idx):
        return self._step(batch,batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch,batch_idx, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
