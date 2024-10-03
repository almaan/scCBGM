import matplotlib.pyplot as plt
import os
import seaborn as sns
import scanpy as sc
import numpy as np
import wandb
import pandas as pd
import rapids_singlecell as rsc



def create_plot_path(original_path, cfg):
    plotting_folder_path = original_path + cfg.plotting.plot_path
    if not os.path.exists(plotting_folder_path):
        os.makedirs(plotting_folder_path)
    return plotting_folder_path


def plot_generation(adata, plotting_folder_path, cfg):

    rsc.pp.pca(adata)
    rsc.pp.neighbors(adata)
    rsc.tl.umap(adata)

    colors = ["ident", "tissue", "celltype", "batch"]

    old_obs = adata.obs.copy()

    if "concepts" in adata.obsm:
        concepts = adata.obsm['concepts'].copy()
        columns = [f'concept_{k}' for k in range(concepts.shape[1])]
        concepts = pd.DataFrame(concepts,
                                index = adata.obs.index,
                                columns = columns,
                                )

        adata.obs = pd.concat(( old_obs, concepts ), axis =1)

        colors += columns

    fig = sc.pl.umap(adata, color=colors, ncols=len(colors), return_fig=True)

    plot_filename = plotting_folder_path + cfg.wandb.experiment + ".png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()

    adata.obs = old_obs

    return plot_filename


def plot_concept_shift(
    results, concept_name, intervention_type, plotting_folder_path, cfg
):
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        results,
        y="values",
        x="coef_direction",
        hue="data",
        split=True,
        gap=0.1,
        inner="quart",
    )

    title = concept_name + " "
    if intervention_type == "On":
        title += ": 0->1"
    else:
        title += ": 1->0"

    ax.set_title(title)

    plot_filename = (
        plotting_folder_path
        + cfg.wandb.experiment
        + "_"
        + concept_name
        + "_turn"
        + intervention_type
        + ".png"
    )
    plt.savefig(plot_filename)
    plt.show()
    plt.close()


def plot_performance_abundance_correlation(
    df_long_global,
    on_off_df,
):

    cp_scores = (
        df_long_global[["concept", "metric", "value"]]
        .groupby(["metric", "concept"])
        .agg("mean")
    )

    x_ratio = on_off_df["1"] / (on_off_df["0"].values + on_off_df["1"].values)
    y_auroc = cp_scores.loc["auroc_joint"]
    inter_auroc = on_off_df.index.intersection(y_auroc.index)
    y_auroc = y_auroc.loc[inter_auroc].values.flatten().tolist()

    y_auprc = cp_scores.loc["auprc_joint"]
    inter_auprc = on_off_df.index.intersection(y_auprc.index)
    y_auprc = y_auprc.loc[inter_auprc].values.flatten().tolist()

    data_auroc = list(
        zip(x_ratio.loc[inter_auroc].values, y_auroc, inter_auroc.tolist())
    )
    data_auprc = list(
        zip(x_ratio.loc[inter_auprc].values, y_auprc, inter_auprc.tolist())
    )

    for data, name in zip([data_auroc, data_auprc], ["AUROC", "AUPRC"]):

        wandb.log(
            {
                f"{name}_vs_ratio": wandb.plot.scatter(
                    wandb.Table(
                        data=data, columns=["#On / (#On+#Off)", name, "concept"]
                    ),
                    "#On / (#On+#Off)",
                    name,
                    title=f"{name}_vs_ratio",
                )
            }
        )

    return None


def plot_concept_correlation_matrix(coefs):

    cmat = coefs.T.corr()

    wandb.log(
        {
            "concept_correlation": wandb.plots.HeatMap(
                cmat.index.tolist(),
                cmat.columns.tolist(),
                cmat.values,
                show_text=True,
            )
        }
    )


def plot_celltype_correlation_matrix(dataset):

    cmat = dataset.celltype_coef.to_dataframe().unstack()["celltype_coef"].T.corr()

    wandb.log(
        {
            "celltype_correlation": wandb.plots.HeatMap(
                cmat.index.tolist(),
                cmat.columns.tolist(),
                cmat.values,
                show_text=True,
            )
        }
    )


def plot_intervention_effect_size(
    effect_size,
):

    for k, (direction, values) in enumerate(effect_size.items()):

        if direction == "neg":
            plot_values = [[-x] for x in values]
        else:
            plot_values = [[x] for x in values]

        wandb.log(
            {
                f"effect_sizes_{direction}": wandb.plot.histogram(
                    wandb.Table(data=plot_values, columns=["cohens_d"]),
                    "cohens_d",
                    title=f"Effect Size : {direction}",
                    num_bins=50,
                )
            }
        )
