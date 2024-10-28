import conceptlab as clab
import numpy as np
import xarray as xr


import omegaconf

import pandas as pd
import wandb
import os
from lightning.pytorch.loggers import WandbLogger
from hydra.utils import get_original_cwd
import pytorch_lightning as pl
import anndata as ad
import torch

from conceptlab.utils.seed import set_seed
from conceptlab.utils import helpers
from conceptlab.utils import plot
from conceptlab.datagen import modify
from conceptlab.utils import constants as C
import conceptlab.evaluation.integration as ig
import conceptlab.evaluation.generation as gen
import conceptlab.evaluation.concepts as con
from omegaconf import DictConfig

import os


import hydra
from omegaconf import DictConfig

MODS = {
    "drop": modify.drop_concepts,
    "add": modify.add_concepts,
    "noise": modify.add_noise,
    "duplicate": modify.add_duplicate,
}


@hydra.main(config_path="./hydra_config/", config_name="config.yaml")
def main(
    cfg: DictConfig,
) -> None:

    set_seed(cfg.constants.seed)
    original_path = get_original_cwd()
    cfg.constants.checkpoint_dir
    cfg.model.get("normalize", True)

    adata_path = (
        original_path + cfg.constants.data_path + cfg.dataset.dataset_name + ".h5ad"
    )

    if not os.path.exists(adata_path):
        raise Exception("Path {} does not exist".format(adata_path))

    adata = ad.read_h5ad(adata_path)
    adata.obsm[cfg.dataset.concept_key]
    adata.varm[cfg.dataset.coefs_key]

    # Split in to train and test

    if cfg.dataset.split_col is None:
        adata_test, adata_train, _, _ = helpers.stratified_adata_train_test_split(adata)
    else:
        pass

    wandb.finish()


if __name__ == "__main__":
    main()
