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
import pickle
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
    checkpoint_dir = osp.join(original_path, cfg.constants.checkpoint_dir)
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

    train_test_col = cfg.dataset.get("train_test_col", None)

    if train_test_col is not None:
        logger.info(f"Splitting w.r.t column {train_test_col}")
        test_labels = cfg.dataset.get("test_labels")
        if test_labels is not None:
            logger.info(f"Test Labels: {test_labels}")
        adata_test, adata, n_test, idx = helpers.controlled_adata_train_test_split(
            adata,
            split_col=train_test_col,
            test_labels=test_labels,
        )

    else:
        adata_test, adata, n_test, idx = helpers.stratified_adata_train_test_split(
            adata,
            concept_key=concept_key,
        )

    logger.info("Anndata Information")

    n_obs, n_vars = adata.shape
    n_concepts = adata.obsm[concept_key].shape[1]
    # TODO: check if this makes sense
    n_concepts_to_eval = cfg.dataset.get("n_concepts", n_concepts)

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        cfg.model.input_dim = n_vars
        cfg.model.n_concepts = n_concepts
        model = model_to_call(config=cfg.model)
    except NotImplementedError as e:
        print(f"Error: {e}")

    data_module = clab.data.dataloader.GeneExpressionDataModule(
        adata,
        add_concepts=model.has_concepts,
        concept_key=concept_key,
        batch_size=512,
        normalize=normalize,
    )

    callbacks = []

    if cfg.save_checkpoints:
        if not osp.isdir(checkpoint_dir):
            logger.info(f"Creating checkpoint directory >>> {checkpoint_dir}")
            os.makedirs(checkpoint_dir)
        else:
            logger.info(f"Checkpoints are saved to >>> {checkpoint_dir}")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=cfg.wandb.experiment,
            save_top_k=1,
            verbose=True,
            monitor="val_Total_loss",
            mode="min",
            every_n_epochs=10,
        )

        callbacks.append(checkpoint_callback)

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
    trainer.fit(model, data_module)

    logger.info("Activate Eval Mode and move to CPU")
    model.to("cpu")
    model.eval()

    if hasattr(model, "log_parameters"):
        model.log_parameters()

    x_raw = adata_test.X.astype(np.float32).copy()
    x_true = adata_test.X.astype(np.float32).copy()
    c_true = adata_test.obsm["concepts"].values.copy().astype(np.float32)

    if normalize:
        logger.info("Normalize data")
        x_true = x_true / x_true.sum(axis=1, keepdims=True) * 1e4
        x_true = np.log1p(x_true)

    sub_idx = np.random.choice(
        x_true.shape[0], replace=False, size=min(5000, x_true.shape[0])
    )

    ad_true = ad.AnnData(
        x_true[sub_idx],
        obs=adata_test.obs.iloc[sub_idx],
    )
    x_concepts = adata_test.obsm[concept_key].copy()

    if cfg.model.type == "CVAE":
        if cfg.given_gt:
            preds = model(torch.tensor(x_true), torch.tensor(c_true))
            c_mean = None
        else:
            c_mean = np.mean(
                adata_train.obsm["concepts"].values.copy().astype(np.float32),
                axis=0,
                keepdims=True,
            )
            # Determine the number of times to repeat (x times)
            x = c_true.shape[0]
            # Repeat the (1, 8) tensor to match the shape of (x, 8)
            c_mean = np.tile(c_mean, (x, 1))
            # print("random")
            c_mean = pd.DataFrame(
                c_mean,
                index=x_concepts.index,
                columns=x_concepts.columns,
            )

            print(c_mean)

            preds = model(helpers._to_tensor(x_true), helpers._to_tensor(c_mean))

    else:
        c_mean = None
        preds = model(torch.tensor(x_true))

    x_pred = preds["x_pred"].detach().numpy()

    if cfg.model.has_cbm:
        pred_concept = preds["pred_concept"].detach().numpy()
        concept_loss_dict = con.concept_accuarcy(
            x_concepts.values,
            pred_concept,
            debug=cfg.DEBUG,
        )
        for key, value in concept_loss_dict.items():
            wandb.log({f"{key}": value})

    ad_pred = ad.AnnData(
        x_pred[sub_idx],
        obs=adata_test.obs.iloc[sub_idx],
    )

    mse_loss = gen.mse_loss(
        x_true, x_pred, normalize_true=(not normalize), normalize_pred=(not normalize)
    )

    r2_score = gen.r2_score(x_true, x_pred)

    wandb.log({"test_MSE_loss": mse_loss})
    wandb.log({"test_r2_score": r2_score})

    merge_dict = dict()
    if cfg.model.has_cbm:
        c_pred = preds["pred_concept"].detach().numpy()
        ad_pred.obsm[concept_key] = pd.DataFrame(
            c_pred[sub_idx], index=x_concepts.index[sub_idx], columns=x_concepts.columns
        )
        ad_true.obsm[concept_key] = x_concepts.iloc[sub_idx]

        if cfg.model.independent_training:
            x_pred_withGT = model(
                helpers._to_tensor(x_true), helpers._to_tensor(x_concepts)
            )["x_pred"]
            x_pred_withGT = x_pred_withGT.detach().numpy()

            ad_pred_withGT = ad.AnnData(
                x_pred_withGT[sub_idx],
                obs=adata_test.obs.iloc[sub_idx],
            )

            ad_pred_withGT.obsm[concept_key] = x_concepts.iloc[sub_idx]
            merge_dict["vae_cbm_withGT"] = ad_pred_withGT

            mse_loss_withGT = gen.mse_loss(
                x_true,
                x_pred_withGT,
                normalize_true=(not normalize),
                normalize_pred=(not normalize),
            )
            wandb.log({"test_mse_loss_withGT": mse_loss_withGT})
    else:
        ad_pred.obsm[concept_key] = x_concepts.iloc[sub_idx].copy()
        ad_true.obsm[concept_key] = x_concepts.iloc[sub_idx].copy()

    merge_dict[f"{cfg.model.type}"] = ad_pred
    merge_dict["true"] = ad_true
    ad_merge = ad.concat(
        merge_dict,
        axis=0,
        label="ident",
    )

    ad_merge.obsm[concept_key] = pd.DataFrame(
        ad_merge.obsm[concept_key],
        index=ad_merge.obs_names,
        columns=adata_test.obsm[concept_key].columns,
    )

    ad_merge.obs_names_make_unique()

    if cfg.plotting.plot:

        plotting_folder_path = plot.create_plot_path(original_path, cfg)

        plot_filename = plot.plot_generation(
            ad_merge, plotting_folder_path, cfg=cfg, concept_key=concept_key
        )
        # Log the plot to wandb
        wandb.log({"generation_plot": wandb.Image(plot_filename)})

    if "test_integration" in cfg:
        ig_test_funs = dict(
            lisi=ig.lisi,
            modularity=ig.modularity,
        )

        if isinstance(cfg.test_integration, str):
            ig_tests = [cfg.test_integration]
        else:
            ig_tests = cfg.test_integration

        for ig_test in ig_tests:
            if ig_test in ig_test_funs:
                ig_scores = ig_test_funs[ig_test](ad_merge, label="ident")
                for key, val in ig_scores.items():
                    wandb.log({f"{ig_test}_{key}": val})

    if model.has_concepts and cfg.test_intervention:

        indicator = adata_test.uns["concept_indicator"]
        ix_og_concepts = indicator.values == C.Mods.none
        original_concepts = adata_test.obsm[concept_key].iloc[:, ix_og_concepts].copy()
        concept_names = original_concepts.columns

        test_index = adata_test.obs_names
        original_test_concepts = original_concepts.loc[test_index]
        coefs = adata.varm[concept_coef_key].T

        true_data = pd.DataFrame(x_true, columns=coefs.columns)
        genetrated_data = pd.DataFrame(x_pred, columns=coefs.columns)
        n_concepts = len(original_test_concepts.columns)

        intervention_scores = dict(On=dict(), Off=dict())
        intervention_data = dict(On=dict(), Off=dict())

        logger.info("Evaluate Interventions")

        test_concepts = cfg.dataset.get("test_concepts")
        if test_concepts is not None:
            concept_ivn_name = test_concepts
            concept_ivn_num = np.array(
                [
                    np.argmax(x == concept_names)
                    for x in test_concepts
                    if x in concept_names
                ]
            )
        else:
            concept_ivn_name = concept_names[0:n_concepts_to_eval]

        for concept_name in concept_ivn_name:
            concept_vars = dict()

            concept_vars["pos"] = coefs.columns[(coefs.loc[concept_name, :] > 0).values]
            concept_vars["neg"] = coefs.columns[(coefs.loc[concept_name, :] < 0).values]
            concept_vars["neu"] = coefs.columns[
                (coefs.loc[concept_name, :] == 0).values
            ]
            concept_vars["all"] = coefs.columns

            interventions_type = ["On", "Off"]
            for _, intervention_type in enumerate(interventions_type):
                (
                    results,
                    scores,
                    perturbed_data,
                ) = clab.evaluation.interventions.eval_intervention(
                    intervention_type,
                    concept_name,
                    x_concepts,
                    x_true,
                    ix_og_concepts,
                    original_test_concepts,
                    true_data,
                    genetrated_data,
                    coefs,
                    concept_vars,
                    model,
                    cfg,
                    c_mean,
                )

                if cfg.plotting.plot:
                    plot.plot_concept_shift(
                        results,
                        concept_name,
                        intervention_type,
                        plotting_folder_path,
                        cfg,
                    )

                if (scores is not None) and (concept_name in scores):
                    intervention_scores[intervention_type][concept_name] = scores[
                        concept_name
                    ]
                    intervention_data[intervention_type][concept_name] = perturbed_data

        joint_score = clab.evaluation.interventions.score_intervention(
            metrics=[
                # "auroc",
                # "auprc",
                "acc",
            ],
            scores=intervention_scores,
            data=intervention_data,
            concept_coefs=coefs,
            concepts=x_concepts,
            raw_data=x_raw,
            n_concepts=n_concepts_to_eval,
            debug=cfg.DEBUG,
            plot=True,
        )

        for key, val in joint_score.items():
            wandb.log({key.upper(): val})

        if cfg.plotting.plot:

            logger.info("Plot reconstruction UMAP")
            helpers.create_composite_image(
                plotting_folder_path + cfg.wandb.experiment,
                plotting_folder_path
                + cfg.wandb.experiment
                + "_intervention_results.png",
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

        if cfg.DEBUG:
            logger.info("Enter DEBUG Mode")
            if not os.path.exists(original_path + "/results/"):
                os.makedirs(original_path + "/results/")

            df = pd.DataFrame(coefs)
            df.to_csv(
                original_path
                + "/results/"
                + cfg.wandb.experiment
                + "_concept_coef.csv",
                index=False,
            )
            clab.utils.helpers.write_dict_to_csv(
                intervention_scores["On"],
                original_path + "/results/" + cfg.wandb.experiment + "_on.csv",
            )
            clab.utils.helpers.write_dict_to_csv(
                intervention_scores["Off"],
                original_path + "/results/" + cfg.wandb.experiment + "_off.csv",
            )

            for _adata, split in zip([adata, adata_test], ["train", "test"]):

                n_on_concepts = _adata.obsm[concept_key].sum(axis=0)
                n_off_concepts = n_obs - n_on_concepts
                concept_names = _adata.obsm[concept_key].columns
                on_off_df = pd.DataFrame(
                    {
                        "1": n_on_concepts.tolist(),
                        "0": n_off_concepts.tolist(),
                    },
                    index=concept_names,
                )

                on_off_df["concept"] = on_off_df.index
                on_off_table = wandb.Table(dataframe=on_off_df)
                wandb.log({f"[{split}] | concept table": on_off_table})

    wandb.finish()


if __name__ == "__main__":
    main()
