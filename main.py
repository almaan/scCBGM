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
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
import anndata as ad
import xarray as xr
import torch
import scanpy as sc
from conceptlab.utils.seed import set_seed
from conceptlab.utils import helpers


from omegaconf import DictConfig, OmegaConf


import os


import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./hydra_config/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    set_seed(cfg.constants.seed)
    original_path = get_original_cwd()
    checkpoint_dir = cfg.constants.checkpoint_dir

    if not os.path.exists(original_path + cfg.constants.data_path):
        os.makedirs(original_path + cfg.constants.data_path)

    adata_path = (
        original_path + cfg.constants.data_path + cfg.dataset.dataset_name + ".h5ad"
    )

    if not os.path.exists(adata_path):
        dataset = clab.datagen.omics.OmicsDataGenerator.generate(
            n_obs=cfg.dataset.n_obs,
            n_vars=cfg.dataset.n_vars,
            n_tissues=cfg.dataset.n_tissues,
            n_celltypes=cfg.dataset.n_celltypes,
            n_batches=cfg.dataset.n_batches,
            n_concepts=cfg.dataset.n_concepts,
        )

    adata = helpers.dataset_to_anndata(dataset, adata_path)
    adata_test, adata = helpers.simple_adata_train_test_split(adata)

    if cfg.model.has_cbm:
        data_module = clab.data.dataloader.GeneExpressionDataModule(
            adata, add_concepts=True, batch_size=512
        )

    else:
        data_module = clab.data.dataloader.GeneExpressionDataModule(
            adata, batch_size=512
        )

    n_obs, n_vars = adata.shape

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        model = model_to_call(
            input_dim=n_vars,
            hidden_dim=cfg.model.hidden_dim,  # these numbers are quite arbitrary
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta,
            learning_rate=cfg.model.lr,
        )
    except NotImplementedError as e:
        print(f"Error: {e}")

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,  # group runs in "MNIST" project
        name=cfg.wandb.experiment,
        log_model="all",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.wandb.experiment,
        save_top_k=1,
        verbose=True,
        monitor="val_Total_loss",
        mode="min",
        every_n_epochs=10,
    )

    max_epochs = cfg.trainer.max_epochs

    trainer = pl.Trainer(
        max_epochs=max_epochs, callbacks=[checkpoint_callback], logger=wandb_logger
    )

    trainer.fit(model, data_module)

    if cfg.plotting.plot:

        model.to("cpu")
        model.eval()

        x_true = adata_test.X.astype(np.float32).copy()
        x_true = x_true / x_true.sum(axis=1, keepdims=True) * 1e4
        x_true = np.log1p(x_true)

        sub_idx = np.random.choice(x_true.shape[0], replace=False, size=2000)

        ad_true = ad.AnnData(
            x_true[sub_idx],
            obs=adata_test.obs.iloc[sub_idx],
        )

        x_pred = model(torch.tensor(x_true))["x_pred"]
        x_pred = x_pred.detach().numpy()

        ad_pred = ad.AnnData(
            x_pred[sub_idx],
            obs=adata_test.obs.iloc[sub_idx],
        )

        if cfg.model.has_cbm:

            x_concepts = adata_test.obsm["concepts"].astype(np.float32).copy()

            x_pred_withGT = model(torch.tensor(x_true), torch.tensor(x_concepts))[
                "x_pred"
            ]
            x_pred_withGT = x_pred_withGT.detach().numpy()

            ad_pred_withGT = ad.AnnData(
                x_pred_withGT[sub_idx],
                obs=adata_test.obs.iloc[sub_idx],
            )

            ad_merge = ad.concat(
                dict(vae_cbm=ad_pred, vae_cbm_withGT=ad_pred_withGT, true=ad_true),
                axis=0,
                label="ident",
            )
        else:
            ad_merge = ad.concat(dict(vae=ad_pred, true=ad_true), axis=0, label="ident")

        ad_merge.obs_names_make_unique()

        sc.pp.pca(ad_merge)
        sc.pp.neighbors(ad_merge)
        sc.tl.umap(ad_merge)

        sc.pl.umap(ad_merge, color=["ident", "tissue", "celltype", "batch"], ncols=4)

        plotting_folder_path = original_path + cfg.plotting.plot_path
        if not os.path.exists(plotting_folder_path):
            os.makedirs(plotting_folder_path)

        plot_filename = plotting_folder_path + cfg.wandb.experiment + ".png"
        plt.savefig(plot_filename)

        # Log the plot to wandb
        wandb.log({"generation_plot": wandb.Image(plot_filename)})

    wandb.finish()


if __name__ == "__main__":
    main()
