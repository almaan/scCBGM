import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch as t


# Define the VAE model
class VAE(pl.LightningModule):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.input_dim = self.input_dim
        self.hidden_dim = self.hidden_dim
        self.latent_dim = self.latent_dim
        self.learning_rate = self.learning_rate

        # Encoder
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.beta = beta

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

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        return {'x_pred':x_pred, 'z':z, 'mu':mu, 'logvar':logvar}

    def loss_function(self, x, x_pred, mu, logvar,**kwargs):
        loss_dict ={}
        MSE = F.mse_loss(x_pred, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss_dict['rec_loss']= MSE
        loss_dict['KL_loss']= KLD
        loss_dict['Total_loss']= MSE + self.beta * KLD 

        return loss_dict



    def _step(self,batch,batch_idx, prefix = 'train'):
        x = batch
        out = self(x)
        loss_dict = self.loss_function(x,**out)

        for key, value in loss_dict.items():
            self.log(f'{prefix}_{key}', value)
        return loss_dict['Total_loss']

    def training_step(self, batch, batch_idx):
        return self._step(batch,batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch,batch_idx, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
