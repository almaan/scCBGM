import conceptlab as clab
import numpy as np
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
import conceptlab.evaluation.interventions as interventions

from omegaconf import DictConfig
from conceptlab.utils import logging
import os.path as osp

import os


import hydra
from omegaconf import DictConfig

MODS = {
    "drop": modify.drop_concepts,
    "add": modify.add_concepts,
    "noise": modify.add_noise,
    "duplicate": modify.add_duplicate,
    "identity": modify.identity,
    "default": modify.identity,
}


@hydra.main(config_path="./synth_hydra_config/", config_name="config.yaml")
def main(
    cfg: DictConfig,
) -> None:

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb_name = cfg.wandb.experiment + "_" + helpers.timestamp()
    wandb_entity = cfg.wandb.get("entity", None)
    run = wandb.init(project=cfg.wandb.project, name=wandb_name, entity=wandb_entity)

    logger = logging.setup_logger()

    logger.info(":zap: Initate Program :zap:")

    set_seed(cfg.constants.seed)
    original_path = get_original_cwd()
    osp.join(original_path, cfg.constants.checkpoint_dir)
    normalize = cfg.dataset.get("normalize", True)
    concept_key = cfg.dataset.get("concept_key", "concepts")
    data_dir = osp.join(original_path, cfg.constants.data_path)

    wandb_logger = WandbLogger(
        log_model=False,
    )

    if not os.path.exists(data_dir):
        logger.info(" :sparkles: Creating Data Directory :point_right: {}")
        os.makedirs(data_dir)

    adata_path = cfg.dataset.dataset_path
    adata = ad.read_h5ad(adata_path)

    if normalize:
        helpers.normalize_counts(adata)

    concept_key = cfg.intervention.concept_key
    logger.info("using concept key: {}".format(concept_key))
    concepts = adata.obsm[concept_key].copy()

    mod = cfg.modify.mod
    if mod != "identity":
        mod_fun = MODS[mod]
        mod_prms = cfg.modify.params
        logger.info("Modification: {}".format(mod))
        logger.info("Modification Parameters: {}".format(mod_prms))

        target_concept_values = concepts[cfg.intervention.target_concept].copy()
        concepts.drop(columns=cfg.intervention.target_concept, inplace=True)

        concepts, _ = mod_fun(concepts=concepts, **mod_prms)
        concepts[cfg.intervention.target_concept] = target_concept_values
        adata.obsm[concept_key] = concepts

    adata_train, adata_ivn, adata_test = helpers.predefined_adata_train_test_split(
        adata,
        split_col=cfg.intervention.split_col,
        test_label=cfg.intervention.hold_out_label,
        ivn_label=cfg.intervention.ivn_label,
        drop_label=cfg.intervention.get("drop_label"),
    )

    del adata

    logger.info("Anndata Information")

    n_obs, n_vars = adata_train.shape
    n_concepts = adata_train.obsm[concept_key].shape[1]
    print(adata_train)
    print(adata_train.obsm[concept_key])

    cfg.model.input_dim = n_vars
    cfg.model.n_concepts = n_concepts

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        if model_to_call is None:
            raise NotImplementedError(
                f"Model {cfg.model.type} not found in clab.models"
            )
        model = model_to_call(config=cfg.model)

        print(cfg.model)
        model = model_to_call(config=cfg.model)
    except NotImplementedError as e:
        print(f"Error: {e}")

    callbacks = []

    callbacks = callbacks if len(callbacks) > 0 else None
    cfg.trainer.max_epochs
    logger.info("Model>>")
    print(model)
    logger.info("Fitting Model")

    logger.info("Activate Eval Mode and move to CPU")

    align_on = cfg.dataset.get("align_on", None)

    if align_on is not None:
        print("Aligning on {}".format(align_on))
        ivn_ix = adata_ivn.obs[align_on].values
        test_ix = adata_test.obs[align_on].values

        ivn_order = np.argsort(ivn_ix)
        test_order = np.argsort(test_ix)

        adata_ivn = adata_ivn[ivn_order].copy()
        adata_test = adata_test[test_order].copy()

    x_ivn = adata_ivn.X.astype(np.float32).copy()
    x_test = adata_test.X.astype(np.float32).copy()
    x_train = adata_train.X.astype(np.float32).copy()

    c_ivn = adata_ivn.obsm[concept_key].values.copy().astype(np.float32)
    c_test = adata_test.obsm[concept_key].values.copy().astype(np.float32)
    c_train = adata_train.obsm[concept_key].values.copy().astype(np.float32)

    print(c_test)

    model.train_loop(
        data=helpers._to_tensor(x_train),
        concepts=helpers._to_tensor(c_train),
        num_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.trainer.get("batch_size", 512),
        num_workers=cfg.trainer.get("num_workers", 0),
    )

    model.eval()

    device = next(model.parameters()).device
    x_ivn_input = helpers._to_tensor(x_ivn, device=device)
    preds_rec = model(x_ivn_input)
    x_ivn_pred = preds_rec["x_pred"].cpu().detach().numpy()

    mse_rec = gen.mse_loss(
        x_ivn,
        x_ivn_pred,
    )
    cosine_sim_rec = gen.cosine_similarity(x_ivn, x_ivn_pred)
    corr_rec = gen.rowwise_correlation(x_ivn, x_ivn_pred)

    wandb.log({"MSE_reconstructed": mse_rec})
    wandb.log({"cosine_sim_reconstructed": cosine_sim_rec})
    wandb.log({"corr_reconstructed": corr_rec})

    if model.has_concepts and cfg.test_intervention:

        concept_ix = np.argmax(
            adata_ivn.obsm[concept_key].columns == cfg.intervention.target_concept
        )

        x_test_pred = clab.evaluation.interventions.eval_intervention(
            model,
            cfg,
            c_ivn=c_ivn,
            x_ivn=x_ivn,
            concept_ix=concept_ix,
            reference_value=cfg.intervention.reference_value,
        )
        mse_ivn = gen.mse_loss(x_test, x_test_pred)
        cosine_sim_ivn = gen.cosine_similarity(x_test, x_test_pred)
        corr_ivn = gen.rowwise_correlation(x_test, x_test_pred)

        wandb.log({"MSE_intervened": mse_ivn})
        wandb.log({"cosine_sim_intervened": cosine_sim_ivn})
        wandb.log({"corr_intervened": corr_ivn})

    # baselines

    print("ivn")
    print(x_train)
    print("true")
    print(x_test)
    print("pred")
    print(x_test_pred)

    mean_ivn = np.mean(x_ivn, axis=0, keepdims=True)
    mean_train = np.mean(x_train, axis=0, keepdims=True)

    mse_mean_ivn = gen.mse_loss(x_test, mean_ivn)
    mse_mean_train = gen.mse_loss(x_test, mean_train)

    wandb.log({"MSE_baseline_mean_intervention_set": mse_mean_ivn})
    wandb.log({"MSE_baseline_mean_train_set": mse_mean_train})

    wandb.finish()


if __name__ == "__main__":
    main()
