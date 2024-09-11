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
from conceptlab.datagen import modify
from conceptlab.utils import constants as C


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

    generate_data = cfg.get("generate_data", False)

    if not os.path.exists(adata_path) or generate_data:

        dataset = clab.datagen.omics.OmicsDataGenerator.generate(
            n_obs=cfg.dataset.n_obs,
            n_vars=cfg.dataset.n_vars,
            n_tissues=cfg.dataset.n_tissues,
            n_celltypes=cfg.dataset.n_celltypes,
            n_batches=cfg.dataset.n_batches,
            n_concepts=cfg.dataset.n_concepts,
        )

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

    try:
        model_to_call = getattr(clab.models, cfg.model.type, None)
        cfg.model.input_dim = cfg.dataset.n_vars
        cfg.model.n_concepts = cfg.dataset.n_concepts
        model = model_to_call(config=cfg.model)
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
            if cfg.model.independent_training:
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
                ad_merge = ad.concat(
                    dict(vae_cbm=ad_pred, true=ad_true),
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
        plt.show()
        plt.close()

        # Log the plot to wandb
        wandb.log({"generation_plot": wandb.Image(plot_filename)})

    if cfg.model.has_cbm and cfg.test_intervention:
        orignal_concepts = dataset.concepts.to_dataframe().unstack()["concepts"].copy()
        test_index = ["obs_" + str(i) for i in idx[0:n_test]]
        orignal_test_concepts = orignal_concepts.loc[test_index]
        coefs = dataset.concept_coef.to_dataframe().unstack()["concept_coef"]
        genetrated_data = pd.DataFrame(x_pred, columns=coefs.columns)
        n_concepts = len(orignal_test_concepts.columns)

        intervention_on_score = np.zeros((n_concepts, 3))
        intervention_off_score = np.zeros((n_concepts, 3))
        strict_intervention_score = 0

        for c, concept_name in enumerate(orignal_test_concepts.columns):
            pos_concept_vars = coefs.columns[(coefs.iloc[c, :] > 0).values]
            neg_concept_vars = coefs.columns[(coefs.iloc[c, :] < 0).values]

            mask = np.zeros_like(x_concepts)
            mask[:, c] = 1

            # Turn concept on ..
            x_concepts_intervene = x_concepts.copy()
            x_concepts_intervene[:, c] = 1

            x_pred_withIntervention = model.intervene(
                torch.tensor(x_true),
                torch.tensor(x_concepts_intervene),
                torch.tensor(mask),
            )["x_pred"]
            x_pred_withIntervention = x_pred_withIntervention.detach().numpy()

            genetrated_data_after_intervention = pd.DataFrame(
                x_pred_withIntervention, columns=coefs.columns
            )

            results = dict(values=[], data=[], coef_direction=[])
            for data_name, data in zip(
                ["perturbed", "original"],
                [genetrated_data_after_intervention, genetrated_data],
            ):
                for direction_name, genes in zip(
                    ["up", "down"], [pos_concept_vars, neg_concept_vars]
                ):
                    ndata = data.loc[:, genes].copy()
                    ndata = ndata.mean(axis=1).values
                    results["values"] += ndata.tolist()
                    results["data"] += len(ndata) * [data_name]
                    results["coef_direction"] += len(ndata) * [direction_name]

            results = pd.DataFrame(results)
            # for visualization
            results["data"] = pd.Categorical(results["data"], ["original", "perturbed"])

            if cfg.plotting.plot:
                ax = sns.violinplot(
                    results,
                    y="values",
                    x="coef_direction",
                    hue="data",
                    split=True,
                    gap=0.1,
                    inner="quart",
                )
                ax.set_title(concept_name + " On")
                plot_filename_turnOn = (
                    plotting_folder_path
                    + cfg.wandb.experiment
                    + "_"
                    + concept_name
                    + "_turnOn.png"
                )
                plt.savefig(plot_filename_turnOn)
                plt.show()
                plt.close()

            on_concepts = pd.DataFrame(
                x_concepts_intervene,
                index=orignal_test_concepts.index,
                columns=orignal_test_concepts.columns,
            )

            eval_res, scores = clab.evaluation.interventions.DistributionShift.score(
                genetrated_data,
                genetrated_data_after_intervention,
                orignal_test_concepts,
                on_concepts,
                coefs,
            )

            intervention_on_score[c, :] = scores
            if scores.sum() == 3:
                strict_intervention_score += 1
            # Turn concept off ..
            x_concepts_intervene = x_concepts.copy()
            x_concepts_intervene[:, c] = 0

            x_pred_withIntervention = model.intervene(
                torch.tensor(x_true),
                torch.tensor(x_concepts_intervene),
                torch.tensor(mask),
            )["x_pred"]
            x_pred_withIntervention = x_pred_withIntervention.detach().numpy()

            genetrated_data_after_intervention = pd.DataFrame(
                x_pred_withIntervention, columns=coefs.columns
            )
            results = dict(values=[], data=[], coef_direction=[])
            for data_name, data in zip(
                ["perturbed", "original"],
                [genetrated_data_after_intervention, genetrated_data],
            ):
                for direction_name, genes in zip(
                    ["up", "down"], [pos_concept_vars, neg_concept_vars]
                ):
                    ndata = data.loc[:, genes].copy()
                    ndata = ndata.mean(axis=1).values
                    results["values"] += ndata.tolist()
                    results["data"] += len(ndata) * [data_name]
                    results["coef_direction"] += len(ndata) * [direction_name]

            results = pd.DataFrame(results)
            # for visualization
            results["data"] = pd.Categorical(results["data"], ["original", "perturbed"])

            if cfg.plotting.plot:
                ax = sns.violinplot(
                    results,
                    y="values",
                    x="coef_direction",
                    hue="data",
                    split=True,
                    gap=0.1,
                    inner="quart",
                )
                ax.set_title(concept_name + " Off")
                plot_filename_turnOff = (
                    plotting_folder_path
                    + cfg.wandb.experiment
                    + "_"
                    + concept_name
                    + "_turnOff.png"
                )
                plt.savefig(plot_filename_turnOff)
                plt.show()
                plt.close()

            off_concepts = pd.DataFrame(
                x_concepts_intervene,
                index=orignal_test_concepts.index,
                columns=orignal_test_concepts.columns,
            )
            eval_res, scores = clab.evaluation.interventions.DistributionShift.score(
                genetrated_data,
                genetrated_data_after_intervention,
                orignal_test_concepts,
                off_concepts,
                coefs,
            )
            intervention_off_score[c, :] = scores
            if scores.sum() == 3:
                strict_intervention_score += 1

        intervention_score = intervention_on_score + intervention_off_score
        intervention_score = intervention_score / 2
        intervention_score = np.mean(intervention_score, axis=0) * 100

        strict_intervention_score = (strict_intervention_score / (2 * n_concepts)) * 100
        if not os.path.exists(original_path + "/results/"):
            os.makedirs(original_path + "/results/")

        concept_names = np.array(orignal_test_concepts.columns).reshape(-1, 1)
        array = np.hstack(
            (concept_names, intervention_on_score, intervention_off_score)
        )

        column_names = [
            "concept name",
            "on_pos",
            "on_neg",
            "on_neu",
            "off_pos",
            "off_neg",
            "off_neu",
        ]
        df = pd.DataFrame(array, columns=column_names)
        df.to_csv(
            original_path + "/results/" + cfg.wandb.experiment + ".csv", index=False
        )

        wandb.log({"intervention pos acc": intervention_score[0]})
        wandb.log({"intervention neg acc": intervention_score[1]})
        wandb.log({"intervention neu acc": intervention_score[2]})
        wandb.log({"strict intervention acc": strict_intervention_score})

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

    wandb.finish()


if __name__ == "__main__":
    main()
