
import sys
SRC_DIR = '../'

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# Import necessary libraries

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
import torch
import os.path as osp

import scanpy as sc


# Set paths

checkpoint_dir = "../data/ckpt/"
data_dir = '../data/demo'
adata_path = os.path.join(data_dir, 'adata-002.h5ad')


# Set variables
GENERATE_DATA = False


# ## Generate Data

np.random.seed(69)

if GENERATE_DATA or not osp.exists(adata_path):

    n_obs = 50000
    n_vars = 5000
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

    adata = ad.AnnData(dataset.data.values,
                       var = pd.DataFrame([], index = dataset['var'].values),
                       obs = pd.DataFrame([], index = dataset['obs'].values),
                      )

    adata.obs['tissue'] = dataset.tissues.values
    adata.obs['celltype'] = dataset.celltypes.values
    adata.obs['batch'] = dataset.batches.values
    adata.obsm['concepts'] = dataset.concepts.values

    adata.write_h5ad(adata_path)


if osp.exists(adata_path):
    adata = ad.read_h5ad(adata_path)


np.random.seeed(69)

idx = np.arange(len(adata))
np.random.shuffle(idx)
p_test = 0.5
n_test = int(0.5 * len(adata))


adata_test, adata = adata[idx[0:n_test]].copy(),adata[idx[n_test::]].copy()

data_module = clab.datagen.dataloader.GeneExpressionDataModule(adata, batch_size=512)


n_obs, n_vars = adata.shape

vae = clab.models.VAE(input_dim=n_vars,
            hidden_dim=128, #these numbers are quite arbitrary
            latent_dim=64,
            beta = 0.01,
            learning_rate=1e-3)


wandb_logger = WandbLogger(project='conceptlab_sync', # group runs in "MNIST" project
                            name="vae_small",
                            log_model='all') 


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='vae_small',
    save_top_k=1,
    verbose=True,
    monitor='val_Total_loss',
    mode='min',
    every_n_epochs = 10, 
    )


max_epochs = 100

trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[
               checkpoint_callback], logger = wandb_logger)

trainer.fit(vae, data_module) 



wandb.finish()



vae.to('cpu')
vae.eval() 


x_true = adata_test.X.astype(np.float32).copy()
x_true = x_true / x_true.sum(axis=1, keepdims = True) * 1e4
x_true = np.log1p(x_true)


x_pred = vae(torch.tensor(x_true))['x_pred']
x_pred = x_pred.detach().numpy()

sub_idx = np.random.choice(x_pred.shape[0], replace = False, size = 2000)


ad_true = ad.AnnData(x_true[sub_idx],
                     obs = adata_test.obs.iloc[sub_idx],
                    )

ad_pred = ad.AnnData(x_pred[sub_idx],
                     obs = adata_test.obs.iloc[sub_idx],
                    )

ad_merge = ad.concat(dict(pred = ad_pred, true = ad_true),axis = 0,label='ident')

ad_merge.obs_names_make_unique()


sc.pp.pca(ad_merge)
sc.pp.neighbors(ad_merge)
sc.tl.umap(ad_merge)

sc.pl.umap(ad_merge, color = ['ident','tissue','celltype','batch'], ncols=1)

