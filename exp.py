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
import conceptlab.preprocess as pp
from conceptlab.datagen import modify
from conceptlab.utils import constants as C
import conceptlab.evaluation.integration as ig
import conceptlab.evaluation.generation as gen
import conceptlab.evaluation.concepts as con
from omegaconf import DictConfig
from conceptlab.utils import logging

import os
import os.path as osp
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="./hydra_config/", config_name="config.yaml")
def main(
    cfg: DictConfig,
) -> None:

    # -- wandb setup -- #
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb_name = cfg.wandb.experiment + "_" + helpers.timestamp()
    wandb_entity = cfg.wandb.get("entity", None)
    run = wandb.init(project=cfg.wandb.project, name=wandb_name, entity=wandb_entity)

    # -- logger setup -- #

    logger = logging.setup_logger()
    logger.info(":zap: Initate Program :zap:")
    wandb_logger = WandbLogger(
        log_model=False,
    )

    # -- general setup -- #

    set_seed(cfg.constants.seed)
    original_path = get_original_cwd()
    checkpoint_dir = cfg.constants.checkpoint_dir
    normalize = cfg.dataset.get("normalize", True)

    concept_key = cfg.dataset.get("concept_key", "concepts")
    cfg.dataset.get("concept_coef_key", None)

    # -- load data -- #

    adata_path = osp.join(original_path, cfg.dataset.path)
    adata = ad.read_h5ad(adata_path)

    if normalize:
        pp.norm.normalize_standard_scanpy(adata)

    # -- train/test setup -- #

    pred_schema = cfg.analysis.pred_schema
    pred_labels = list(pred_schema.keys())
    test_labels = cfg.analysis.test_labels

    if not os.path.exists(adata_path):
        raise Exception("Path {} does not exist".format(adata_path))

    adata_train, adata_test, adata_pred = helpers.custom_adata_train_test_split(
        adata,
        split_col=cfg.analysis.split_col,
        test_labels=test_labels,
        pred_labels=pred_labels,
        split_pred=cfg.analysis.split_pred,
    )
    del adata

    # -- setup model -- #

    input_dim = adata_train.shape[1]
    n_concepts = adata_train.obsm[concept_key].shape[1]

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        model = model_to_call(
            config=cfg.model, input_dim=input_dim, n_concepts=n_concepts
        )
    except NotImplementedError as e:
        print(f"Error: {e}")

    # -- setup dataloader (TRAIN) -- #

    data_module = clab.data.dataloader.GeneExpressionDataModule(
        adata_train,
        add_concepts=model.has_concepts,
        concept_key=concept_key,
        batch_size=cfg.model.get("batch_size", 512),
        normalize=False,
    )

    # -- setup callbacks and trainer -- #

    callbacks = []

    # add checkpoint saving if specified
    if cfg.save_checkpoints:
        if not osp.isdir(checkpoint_dir):
            logger.info(f"Creating checkpoint directory >>> {checkpoint_dir}")
            os.makedirs(checkpoint_dir)
        else:
            logger.info(f"Checkpoints are saved to >>> {checkpoint_dir}")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            verbose=True,
            monitor="val_Total_loss",
            mode="min",
            every_n_epochs=10,
        )

        callbacks.append(checkpoint_callback)

    # add early stopping when specified
    if cfg.model.get("early_stopping", False):
        logger.info("Using Early Stopping")
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_concept_loss",
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    callbacks = callbacks if len(callbacks) > 0 else None

    max_epochs = cfg.trainer.max_epochs
    logger.info("Fitting Model")
    trainer = pl.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=callbacks
    )

    logger.info("Model>>")
    print(model)

    # -- fit model -- #
    trainer.fit(model, data_module)

    logger.info("Activate Eval Mode and move to CPU")

    # -- move model to cpu and eval mode -- #
    model.to("cpu")
    model.eval()

    # -- Get Predictions -- #

    x_train = adata_train.to_df()
    adata_train.obsm[concept_key]
    preds_train = model(helpers._to_tensor(x_train))
    x_train_pred = preds_train["x_pred"].detach().numpy()

    x_pred = adata_pred.to_df()
    c_pred = adata_pred.obsm[concept_key]
    preds_pred = model(helpers._to_tensor(x_pred))
    x_pred_pred = preds_pred["x_pred"].detach().numpy()

    x_test = adata_test.to_df()
    adata_test.obsm[concept_key]
    preds_test = model(helpers._to_tensor(x_test))
    x_test_pred = preds_test["x_pred"].detach().numpy()

    # -- Evaluate Reconcstruction -- #

    wandb.log({"test_MSE_loss": gen.mse_loss(x_test, x_test_pred)})
    wandb.log({"pred_MSE_loss": gen.mse_loss(x_pred, x_pred_pred)})
    wandb.log({"train_MSE_loss": gen.mse_loss(x_train, x_train_pred)})

    for label, schema in pred_schema.items():

        on_concepts = schema.get(1, [])
        off_concepts = schema.get(0, [])

        on_concepts_txt = ",".join(on_concepts)
        off_concepts_txt = ",".join(off_concepts)

        logger.info(
            f"{label} : On :zap: [ {on_concepts_txt} ] | Off :zap: [ {off_concepts_txt} ]"
        )

        x_ivn = clab.evaluation.interventions.intervene(
            x_pred, c_pred, model, on_concepts=on_concepts, off_concepts=off_concepts
        )

        target = schema.get("target", None)

        if target is not None:
            is_target = adata_test.obs[cfg.analysis.split_col] == target
            is_target = None if len(is_target) < 1 else is_target
        else:
            is_target = helpers.find_matching_target(
                on_concepts, off_concepts, target_concepts
            )

        if is_target is not None:
            x_target = adata_test[is_target].to_df()

            target_score = (
                clab.evaluation.interventions.evaluate_intervention_with_target(
                    x_pred,
                    x_ivn,
                    x_target,
                )
            )

            for key, val in target_score.items():
                wandb.log({f"{label}_{key}": val})

    wandb.finish()


if __name__ == "__main__":
    main()
