import matplotlib.pyplot as plt
import os
import seaborn as sns
import scanpy as sc
def create_plot_path(original_path,cfg):
    plotting_folder_path = original_path + cfg.plotting.plot_path
    if not os.path.exists(plotting_folder_path):
        os.makedirs(plotting_folder_path)
    return plotting_folder_path


def plot_generation(adata,plotting_folder_path,cfg):
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color=["ident", "tissue", "celltype", "batch"], ncols=4)



    plot_filename = plotting_folder_path + cfg.wandb.experiment + ".png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
    return plot_filename

def plot_concept_shift(results,concept_name,intervention_type,plotting_folder_path,cfg):
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
    ax.set_title(concept_name + " "+intervention_type)
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