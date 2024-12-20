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
import conceptlab.preprocess as pp
from conceptlab.utils import helpers
from conceptlab.utils import plot
from conceptlab.datagen import modify
from conceptlab.utils import constants as C
from conceptlab.evaluation.interventions import DistributionShift
import conceptlab.evaluation.integration as ig
import conceptlab.evaluation.generation as gen
import conceptlab.evaluation.concepts as con
from omegaconf import DictConfig
import pickle
from conceptlab.utils import logging
import os.path as osp

import os
import biolord


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


@hydra.main(config_path="./dev_hydra_config/", config_name="config.yaml")
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
    normalize = cfg.model.get("normalize", True)
    concept_key = cfg.dataset.get("concept_key", "concepts")
    concept_coef_key = cfg.dataset.get("concept_coef_key", "concept_coef")
    data_dir = osp.join(original_path, cfg.constants.data_path)

    wandb_logger = WandbLogger(
        log_model=False,
    )

    if not os.path.exists(data_dir):
        logger.info(" :sparkles: Creating Data Directory :point_right: {}")
        os.makedirs(data_dir)

    adata_path = osp.join(data_dir, cfg.dataset.dataset_name + ".h5ad")

    if cfg.save_generated_data:
        logger.info(f"AnnData Path :point_right: {adata_path}")

    generate_data = cfg.get("generate_data", True)

    if not os.path.exists(adata_path) or generate_data:
        logger.info(":seedling: Initiated Data Generation :seedling:")
        dataset_path = (
            original_path + cfg.constants.data_path + cfg.dataset.dataset_name + ".pkl"
        )

        dataset = clab.datagen.omics.OmicsDataGenerator.generate(**cfg.dataset)
        logger.info(":evergreen_tree: Completed Data Generation :evergreen_tree:")

        if cfg.dataset.add_all_effects_as_concepts:
            dataset = helpers.add_extras_to_concepts(dataset, cfg)

        if cfg.save_generated_data:
            adata.write_h5ad(adata_path)

            with open(dataset_path, "wb") as file:
                pickle.dump(dataset, file)

        mod = cfg.modify.mod
        mod_fun = MODS[mod]
        mod_prms = cfg.modify.params
        concepts, indicator = mod_fun(dataset=dataset, **mod_prms)
        adata = helpers.dataset_to_anndata(
            dataset,
            concepts=concepts,
            concept_key=concept_key,
        )

        adata.uns["concept_indicator"] = indicator

    else:
        adata = ad.read_h5ad(adata_path)
        adata.X = adata.to_df().values
        adata.uns["concept_indicator"] = np.array(
            [C.Mods.none] * adata.obsm[concept_key].shape[1], dtype="<U64"
        )

    if normalize:
        pp.norm.normalize_standard_scanpy(adata)

    split_idxs = helpers.stratified_adata_train_test_split(
        adata, concept_key=concept_key, return_index_only=True
    )

    train_val_idx = split_idxs["train"]
    np.random.shuffle(train_val_idx)
    n_val = int(0.2 * len(train_val_idx))
    val_idx = train_val_idx[0:n_val]
    test_idx = split_idxs["test"]

    split_col = np.array(["train"] * len(adata), dtype="<U64")
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"

    adata.obs["split_col"] = split_col
    adata.obs = pd.concat((adata.obs, adata.obsm[concept_key]), axis=1)

    observed_concept_names = adata.obsm[concept_key].columns.tolist()

    logger.info("Anndata Information")

    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=None,
        categorical_attributes_keys=observed_concept_names,
    )

    # module_params = {
    #     "decoder_width": cfg.model.decoder_width,
    #     "decoder_depth": cfg.model.decoder_depth,
    #     "attribute_nn_width": cfg.model.attribute_nn_width,
    #     "attribute_nn_depth": cfg.model.attribute_nn_depth,
    #     "n_latent_attribute_categorical": cfg.model.n_latent_attribute_categorical,
    #     "gene_likelihood": cfg.model.gene_likelihood,
    #     "reconstruction_penalty": cfg.model.reconstruction_penalty,
    #     "unknown_attribute_penalty": cfg.model.unknown_attribute_penalty,
    #     "unknown_attribute_noise_param": cfg.model.unknown_attribute_noise_param,
    #     "attribute_dropout_rate": cfg.model.attribute_dropout_rate,
    #     "use_batch_norm": cfg.model.use_batch_norm,
    #     "use_layer_norm": cfg.model.use_layer_norm,
    #     "seed": cfg.constants.seed,
    # }
    module_params = {k: v for k, v in cfg.model.module_params.items()}
    module_params["seed"] = cfg.constants.seed

    model = biolord.Biolord(
        adata=adata,
        n_latent=cfg.model.model.n_latent,
        module_params=module_params,
        train_classifiers=False,
        split_key="split_col",
        train_split="train",
        valid_split="val",
        test_split="test",
    )

    trainer_params = {k: v for k, v in cfg.model.trainer.items()}
    batch_size = trainer_params.pop("batch_size")

    model.train(
        max_epochs=cfg.trainer.max_epochs,
        batch_size=batch_size,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=10,
        num_workers=1,
        enable_checkpointing=False,
    )

    indicator = adata.uns["concept_indicator"]
    ix_og_concepts = indicator.values == C.Mods.none
    original_concepts = adata.obsm[concept_key].iloc[:, ix_og_concepts].copy()
    concept_names = original_concepts.columns
    coefs = adata.varm[concept_coef_key].T

    scores = dict(On={}, Off={})

    for concept_name in concept_names:

        concept_vars = dict()

        concept_vars["pos"] = coefs.columns[(coefs.loc[concept_name, :] > 0).values]
        concept_vars["neg"] = coefs.columns[(coefs.loc[concept_name, :] < 0).values]
        concept_vars["neu"] = coefs.columns[(coefs.loc[concept_name, :] == 0).values]
        concept_vars["all"] = coefs.columns

        for ivn_value, intervention_type in enumerate(["Off", "On"]):

            source_idx = (adata.obs["split_col"].values == "test") & (
                adata.obs[concept_name].values == ivn_value
            )
            adata_source = adata[source_idx]

            adata_preds = model.compute_prediction_adata(
                adata, adata_source, target_attributes=[concept_name]
            )

            x_old = adata_source[
                adata_source.obs[concept_name].values == ivn_value
            ].to_df()
            x_new = adata_preds[
                adata_preds.obs[concept_name].values == 1 - ivn_value
            ].to_df()

            c_old = ivn_value * np.ones(x_old.shape[0])
            c_new = 1 - c_old

            c_old = pd.DataFrame(c_old, columns=[concept_name])
            c_new = pd.DataFrame(c_new, columns=[concept_name])

            d_values = DistributionShift.score(
                x_old,
                x_new,
                c_old,
                c_new,
                concept_coefs=coefs,
                concept_names=concept_name,
                use_neutral=False,
            )

            scores[intervention_type][concept_name] = d_values[concept_name]

        joint_score = clab.evaluation.interventions.score_intervention(
            metrics=[
                "acc",
            ],
            scores=scores,
        )

    for key, val in joint_score.items():
        wandb.log({key.upper(): val})

    if cfg.plotting.plot:

        logger.info("Plot reconstruction UMAP")
        helpers.create_composite_image(
            plotting_folder_path + cfg.wandb.experiment,
            plotting_folder_path + cfg.wandb.experiment + "_intervention_results.png",
        )
        wandb.log(
            {
                "intervention_plot": wandb.Image(
                    plotting_folder_path
                    + cfg.wandb.experiment
                    + "_intervention_results.png"
                )
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
