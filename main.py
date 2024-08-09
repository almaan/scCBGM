
import conceptlab as clab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import argparse
import os
import pathlib
from lightning.pytorch.loggers import WandbLogger
import pytorch_lightning as pl
import anndata as ad

# import clab.datagen.GeneExpressionDataModule

# from models import (
#                 MLM_concept_input_cbm_decoder_orthognal,
#                 )

# model_parser = { 
#                 "MLM_concept_input_cbm_decoder_orthognal":MLM_concept_input_cbm_decoder_orthognal,
#                 }




# def main():


if __name__ == "__main__":


    n_obs = 50000
    n_vars = 1000

    n_tissues = 3
    n_celltypes = 10
    n_batches = 2
    n_concepts = 8

 
    dataset = clab.datagen.omics.OmicsDataGenerator.generate(n_obs = n_obs,
                                                             n_vars = n_vars,
                                                             n_tissues=n_tissues,
                                                             n_celltypes=n_celltypes,
                                                             n_batches = n_batches,
                                                             n_concepts = n_concepts,
                                                            )



    data_module = clab.datagen.dataloader.GeneExpressionDataModule(dataset, batch_size=512) 


    vae = clab.models.VAE(input_dim=n_vars,
                hidden_dim=1024, #these numbers are quite arbitrary
                latent_dim=512,
                beta = 0.01,
                learning_rate=1e-3)


    wandb_logger = WandbLogger(project='conceptlab_sync', # group runs in "MNIST" project
                                name="vae",
                                log_model='all') 



    # data_module = clab.datagen.dataloader.GeneExpressionDataModule(dataset,add_concepts=True, batch_size=512) 

    # vae = clab.models.CB_VAE(input_dim=n_vars,
    #             hidden_dim=512, #these numbers are quite arbitrary
    #             latent_dim=256,
    #             n_concepts=n_concepts,
    #             beta = 1e-3,
    #             learning_rate=1e-4)


    # wandb_logger = WandbLogger(project='conceptlab_sync', # group runs in "MNIST" project
    #                             name="vae_cbm",
    #                             log_model='all') 


    max_epochs = 100
    trainer = pl.Trainer(max_epochs=max_epochs, logger = wandb_logger)

    trainer.fit(vae, data_module) 



    wandb.finish()

