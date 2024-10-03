import conceptlab as clab
import numpy as np


import omegaconf

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

from conceptlab.utils.seed import set_seed
from conceptlab.utils import helpers
from conceptlab.utils import plot
from conceptlab.datagen import modify
from conceptlab.utils import constants as C
import conceptlab.evaluation.integration as ig

from omegaconf import DictConfig, OmegaConf
import pickle


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
    checkpoint_dir = cfg.constants.checkpoint_dir

    if not os.path.exists(original_path + cfg.constants.data_path):
        os.makedirs(original_path + cfg.constants.data_path)

    adata_path = (
        original_path + cfg.constants.data_path + cfg.dataset.dataset_name + ".h5ad"
    )

    dataset_path = (
        original_path + cfg.constants.data_path + cfg.dataset.dataset_name + ".pkl"
    )

    generate_data = cfg.get("generate_data", True)

    if not os.path.exists(adata_path) or generate_data:

        dataset = clab.datagen.omics.OmicsDataGenerator.generate(**cfg.dataset)

        mod = cfg.modify.mod

        if mod is not None:
            mod_fun = MODS[mod]
            mod_prms = cfg.modify.params
            concepts, indicator = mod_fun(dataset=dataset, **mod_prms)
            adata = helpers.dataset_to_anndata(dataset, concepts, adata_path)
            adata.uns["concept_indicator"] = indicator

        else:
            adata = helpers.dataset_to_anndata(dataset, adata_path=adata_path)
            adata.uns["concept_indicator"] = np.array(
                [C.Mods.none] * adata.obsm["concepts"].shape[1], dtype="<U64"
            )

            # Saving the data object
        with open(dataset_path, "wb") as file:
            pickle.dump(dataset, file)

    else:
        adata = ad.read_h5ad(adata_path)
        # Loading the data object
        with open(dataset_path, "rb") as file:
            dataset = pickle.load(file)

    adata_test, adata, n_test, idx = helpers.simple_adata_train_test_split(adata)

    if cfg.model.has_cbm:
        data_module = clab.data.dataloader.GeneExpressionDataModule(
            adata, add_concepts=True, batch_size=512
        )

    else:
        data_module = clab.data.dataloader.GeneExpressionDataModule(
            adata, batch_size=512
        )

    n_obs, n_vars = adata.shape
    n_concepts = adata.obsm["concepts"].shape[1]

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        cfg.model.input_dim = n_vars
        cfg.model.n_concepts = n_concepts
        model = model_to_call(config=cfg.model)
    except NotImplementedError as e:
        print(f"Error: {e}")

    # this part is necessary for sweeps to work
    wandb.finish()
    wandb_name = cfg.wandb.experiment + "_" + helpers.timestamp()

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=wandb_name,
        log_model=False,
    )

    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb_logger.experiment.config.update(wandb_config)

    callbacks = []

    if cfg.save_checkpoints:
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

    callbacks = callbacks if len(callbacks) > 0 else None
    max_epochs = cfg.trainer.max_epochs
    trainer = pl.Trainer(
        max_epochs=max_epochs, logger=wandb_logger, callbacks=callbacks
    )
    trainer.fit(model, data_module)

    if cfg.plotting.plot or cfg.test_intervention:

        model.to("cpu")
        model.eval()

        x_true = adata_test.X.astype(np.float32).copy()
        x_true = x_true / x_true.sum(axis=1, keepdims=True) * 1e4
        x_true = np.log1p(x_true)

        sub_idx = np.random.choice(
            x_true.shape[0], replace=False, size=min(5000, x_true.shape[0])
        )
        # sub_idx = np.arange(x_true.shape[0])

        ad_true = ad.AnnData(
            x_true[sub_idx],
            obs=adata_test.obs.iloc[sub_idx],
        )

        preds = model(torch.tensor(x_true))
        x_pred = preds["x_pred"].detach().numpy()
        x_concepts = adata_test.obsm["concepts"].astype(np.float32).copy()

        ad_pred = ad.AnnData(
            x_pred[sub_idx],
            obs=adata_test.obs.iloc[sub_idx],
        )


    if cfg.plotting:
        plotting_folder_path = plot.create_plot_path(original_path, cfg)
        wandb.log(
            {
                "fc3": wandb.Image(
                    model.state_dict()["fc3.weight"].detach().numpy(), caption="FC3"
                )
            },
        )


        if cfg.model.has_cbm:
            c_pred = preds["pred_concept"].detach().numpy()
            ad_pred.obsm["concepts"] = c_pred[sub_idx]

            ad_true.obsm["concepts"] = x_concepts[sub_idx]
            if cfg.model.independent_training:
                x_pred_withGT = model(torch.tensor(x_true), torch.tensor(x_concepts))[
                    "x_pred"
                ]
                x_pred_withGT = x_pred_withGT.detach().numpy()

                ad_pred_withGT = ad.AnnData(
                    x_pred_withGT[sub_idx],
                    obs=adata_test.obs.iloc[sub_idx],
                )

                ad_pred_withGT.obsm["concepts"] = x_concepts[sub_idx]

                ad_merge = ad.concat(
                    dict(vae_cbm=ad_pred, vae_cbm_withGT=ad_pred_withGT, true=ad_true),
                    axis=0,
                    label="ident",
                )
            else:
                ad_merge = ad.concat(
                    dict(vae_cbm=ad_pred, true=ad_true),
                    axis=0,
                    label="ident",
                )
        else:
            ad_pred.obsm["concepts"] = x_concepts[sub_idx].copy()
            ad_true.obsm["concepts"] = x_concepts[sub_idx].copy()
            ad_merge = ad.concat(dict(vae=ad_pred, true=ad_true), axis=0, label="ident")

        # this scales very poorly
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

        ad_merge.obs_names_make_unique()
        plot_filename = plot.plot_generation(ad_merge, plotting_folder_path, cfg)
        # Log the plot to wandb
        wandb.log({"generation_plot": wandb.Image(plot_filename)})

    if cfg.model.has_cbm and cfg.test_intervention:

        indicator = adata_test.uns["concept_indicator"]
        ix_og_concepts = indicator == C.Mods.none
        original_concepts = adata_test.obsm["concepts"][:, ix_og_concepts].copy()
        original_concepts = pd.DataFrame(
            original_concepts,
            index=adata_test.obs_names,
            columns=[f"concept_{k}" for k in range(original_concepts.shape[1])],
        )

        test_index = ["obs_" + str(i) for i in idx[0:n_test]]
        original_test_concepts = original_concepts.loc[test_index]
        coefs = dataset.concept_coef.to_dataframe().unstack()["concept_coef"]

        true_data = pd.DataFrame(x_true, columns=coefs.columns)
        genetrated_data = pd.DataFrame(x_pred, columns=coefs.columns)
        n_concepts = len(original_test_concepts.columns)

        intervention_scores = dict(On=dict(), Off=dict())
        all_curves = dict(On=dict(), Off=dict())

        for c, concept_name in enumerate(original_test_concepts.columns):
            concept_vars = dict()
            concept_vars["pos"] = coefs.columns[(coefs.iloc[c, :] > 0).values]
            concept_vars["neg"] = coefs.columns[(coefs.iloc[c, :] < 0).values]
            concept_vars["neu"] = coefs.columns[(coefs.iloc[c, :] == 0).values]
            concept_vars["all"] = coefs.columns

            interventions_type = ["On", "Off"]
            for _, intervention_type in enumerate(interventions_type):
                results, scores, curves = (
                    clab.evaluation.interventions.eval_intervention(
                        intervention_type,
                        c,
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
                    )
                )

                if cfg.plotting.plot:
                    plot.plot_concept_shift(
                        results,
                        concept_name,
                        intervention_type,
                        plotting_folder_path,
                        cfg,
                    )

                intervention_scores[intervention_type].update(scores)
                all_curves[intervention_type].update(curves)

        if not os.path.exists(original_path + "/results/"):
            os.makedirs(original_path + "/results/")

        flat_list = helpers.flatten_to_list_of_lists(intervention_scores)
        df_long = pd.DataFrame(flat_list)
        columns = ["intervention", "concept", "direction", "cohens_d"]
        df_long.columns = columns

        df_long.to_csv(
            original_path + "/results/" + cfg.wandb.experiment + ".csv", index=False
        )

        long_global_ix = np.isin(
            df_long["direction"].values, ["auprc_joint", "auroc_joint"]
        )

        df_long_global = df_long.iloc[long_global_ix].copy()
        df_long_global.columns = ["intervention", "concept", "metric", "value"]

        # Convert the DataFrame into a W&B table
        global_table = wandb.Table(dataframe=df_long_global)
        wandb.log({"results global": global_table})

        av_scores = df_long_global[["metric", "value"]].groupby("metric").agg("mean")

        wandb.log({f"AUPRC": av_scores.loc["auprc_joint"].values})
        wandb.log({f"AUROC": av_scores.loc["auroc_joint"].values})

        plot.plot_performance_curves(all_curves)

        if cfg.plotting.plot:

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
            df_long_local = df_long.iloc[~long_global_ix].copy()
            local_table = wandb.Table(dataframe=df_long_local)
            wandb.log({"results local": local_table})

            for _adata, split in zip([adata, adata_test], ["train", "test"]):

                n_on_concepts = _adata.obsm["concepts"].sum(axis=0)
                n_off_concepts = n_obs - n_on_concepts
                on_off_df = pd.DataFrame(
                    {
                        "1": n_on_concepts.tolist(),
                        "0": n_off_concepts.tolist(),
                    },
                    index=[f"concept_{k}" for k in range(n_concepts)],
                )

                on_off_df["concept"] = on_off_df.index
                on_off_table = wandb.Table(dataframe=on_off_df)
                wandb.log({f"[{split}] | concept table": on_off_table})

                plot.plot_performance_abundance_correlation(
                    df_long_global,
                    on_off_df,
                )

                plot.plot_intervention_effect_size(
                    df_long_local,
                )

                # plot.plot_concept_correlation_matrix(coefs)
                # plot.plot_celltype_correlation_matrix(dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
