import matplotlib.pyplot as plt
import os
import seaborn as sns
import scanpy as sc
import wandb
import pandas as pd
from conceptlab.utils import helpers


def create_plot_path(original_path, cfg):
    plotting_folder_path = original_path + cfg.plotting.plot_path
    if not os.path.exists(plotting_folder_path):
        os.makedirs(plotting_folder_path)
    return plotting_folder_path


def plot_generation(
    adata,
    plotting_folder_path,
    cfg,
    normalize=False,
    concept_key="concepts",
    plot_concepts: bool = False,
):

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    colors = [col for col in adata.obs.columns if hasattr(adata.obs[col], "cat")]

    old_obs = adata.obs.copy()

    if plot_concepts:
        if concept_key in adata.obsm:
            concepts = adata.obsm[concept_key].copy()
            columns = ["concept_{}".format(x) for x in concepts.columns.tolist()]
            concepts = pd.DataFrame(
                concepts.values,
                index=adata.obs.index,
                columns=columns,
            )

            adata.obs = pd.concat((old_obs, concepts), axis=1)

            colors += columns

    fig = sc.pl.umap(adata, color=colors, ncols=len(colors), return_fig=True)
    fig.tight_layout()

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
            "concept_correlation": wandb.plot.HeatMap(
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
            "celltype_correlation": wandb.plot.HeatMap(
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

    fig, ax = plt.subplots(1, 1)

    sns.histplot(effect_size, x="cohens_d", hue="direction", ax=ax, bins=25)

    wandb.log({f"effect_sizes": wandb.Image(fig)})


def plot_performance_curves(curve_dict):

    score_names = helpers.get_n_level_keys(curve_dict, 3)
    n_scores = len(score_names)
    score_to_ax = {s: k for k, s in enumerate(score_names)}
    fig, ax = plt.subplots(1, n_scores, figsize=(n_scores * 9, 8))

    df = dict(
        intervention=[],
        concept=[],
        metric=[],
        x=[],
        y=[],
    )

    for i, (iv_key, iv_val) in enumerate(curve_dict.items()):
        for c_key, c_val in iv_val.items():
            for s_key, s_val in c_val.items():
                x = s_val["x"]
                y = s_val["y"]
                n = len(x)

                df["intervention"] += [iv_key] * n
                df["concept"] += [c_key] * n
                df["metric"] += [s_key] * n
                df["x"] += x.tolist()
                df["y"] += y.tolist()

    df = pd.DataFrame(df)

    for s, k in score_to_ax.items():
        sub_df = df.iloc[df["metric"].values == s]
        sns.lineplot(
            sub_df,
            x="x",
            y="y",
            hue="concept",
            style="intervention",
            ax=ax[k],
            errorbar=None,
        )
        ax[k].set_title(s.split("_")[0].upper())

    fig.tight_layout()

    wandb.log({"performance_curves": wandb.Image(fig)})
