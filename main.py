
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
import xarray as xr


if __name__ == "__main__":

    checkpoint_dir = "/data/bucket/ismaia11/conceptlab/models/"

    # n_obs = 50000
    # n_vars = 5000
    # n_tissues = 3
    # n_celltypes = 10
    # n_batches = 2
    # n_concepts = 8

    # dataset = clab.datagen.omics.OmicsDataGenerator.generate(n_obs = n_obs,
    #                                                          n_vars = n_vars,
    #                                                          n_tissues=n_tissues,
    #                                                          n_celltypes=n_celltypes,
    #                                                          n_batches = n_batches,
    #                                                          n_concepts = n_concepts,
    #                                                         )


    # dataset.to_netcdf('./data/complete_set_dataset.nc')


    # data_pth = "./data/5a611776-aae0-41b9-9f2b-aaf5f83771a3.h5ad"

    # adata = ad.read_h5ad(data_pth) 
    # subset_cells = adata[0:50000, :]

    # # subset_cells.write("./data/subset_cells.h5ad")

    adata = ad.read_h5ad(data_pth) 
    subset_cells = adata[0:50000, 0:5000]


    # # print(adata.shape)
    # # print(subset_cells.shape)

    # # print(stop)
    # # data_pth = "./data/subset_cells.h5ad"
    # # dataset = ad.read_h5ad(data_pth) 
    # n_samples,n_vars = subset_cells.shape

    # data_module = clab.datagen.dataloader.GeneExpressionDataModule(subset_cells, batch_size=512) 



    print("Reading")
    dataset = xr.open_dataset('./data/complete_set_dataset.nc')

    print("Creating a data_module")
    data_module = clab.datagen.dataloader.GeneExpressionDataModule(dataset, batch_size=512)



    print("Training")
    vae = clab.models.VAE(input_dim=n_vars,
                hidden_dim=128, #these numbers are quite arbitrary
                latent_dim=64,
                beta = 0.01,
                learning_rate=1e-3)


    wandb_logger = WandbLogger(project='conceptlab_sync', # group runs in "MNIST" project
                                name="vae",
                                log_model='all') 



    


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='vae',
        save_top_k=1,
        verbose=True,
        monitor='val_Total_loss',
        mode='min'
        )





    # data_module = clab.datagen.dataloader.GeneExpressionDataModule(dataset,add_concepts=True, batch_size=512) 


    # print("Training")
    # vae = clab.models.CB_VAE(input_dim=n_vars,
    #             hidden_dim=128, #these numbers are quite arbitrary
    #             latent_dim=64,
    #             n_concepts=n_concepts,
    #             beta = 1e-3,
    #             learning_rate=1e-4)


    # wandb_logger = WandbLogger(project='conceptlab_sync', # group runs in "MNIST" project
    #                             name="vae_cbm",
    #                             log_model='all') 




    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     filename='vae_cbm',
    #     save_top_k=1,
    #     verbose=True,
    #     monitor='val_Total_loss',
    #     mode='min'
    #     )




    max_epochs = 100
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[
                   checkpoint_callback], logger = wandb_logger)

    trainer.fit(vae, data_module) 



    wandb.finish()

