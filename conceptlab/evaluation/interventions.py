from conceptlab.evaluation._base import EvaluationClass
from conceptlab.evaluation import metrics as met
from conceptlab.utils import helpers
import pandas as pd
import plotly.graph_objects as go
from typing import Literal, List, Dict, Any
from torch.nn import Module
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
)

import sklearn.neighbors
import torch
import wandb
from scipy.stats import spearmanr, pearsonr


def eval_intervention(
    model,
    cfg,
    c_ivn,
    x_ivn,
    concept_ix,
    reference_value=0,
):

    # in cbm: input_concept = pred_concepts * (1 - mask) + given_concepts * mask

    c_flip = c_ivn.copy()
    mask = np.zeros_like(c_ivn)

    given_gt = cfg.model.get("given_gt", False)
    if given_gt:
        # we only use the given concepts (mask all predicted)
        mask = np.ones_like(c_ivn)
    else:
        # we use all predicted concepts except the intervened
        mask = np.zeros_like(c_ivn)
        mask[:, concept_ix] = 1

    print("og")
    print(c_flip)
    c_flip[:, concept_ix] = 1 - reference_value
    print("given")
    print(c_flip)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    x_pred = model.intervene(
        x=helpers._to_tensor(x_ivn, device),
        concepts=helpers._to_tensor(c_flip, device),
        mask=helpers._to_tensor(mask, device),
    )["x_pred"]

    x_pred = x_pred.detach().cpu().numpy()

    return x_pred


def compute_auroc(y_true, y_pred, plot=False):
    true_to_val = dict(
        pos=1,
        neg=1,
        neu=0,
    )
    y_true_adj = np.array([true_to_val[x] for x in y_true]).astype(float)
    y_pred_adj = y_pred.copy()
    y_pred_adj[y_true == "neg"] = -y_pred_adj[y_true == "neg"]

    if plot:
        fpr, tpr, _ = roc_curve(y_true_adj, y_pred_adj)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="data"))

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="random",
                line=dict(color="red", dash="solid"),
            )
        )

        fig.update_layout(
            title="ROC",
            xaxis_title="FPR",
            yaxis_title="TPR",
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=[0, 1],
            ),  # Ensure equal scale
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                range=[0, 1],
            ),  # Ensure equal scale
            width=600,
            height=600,
        )

        wandb.log({"roc": fig})

    score = roc_auc_score(
        y_true_adj,
        y_pred_adj,
    )

    return score


def compute_auprc(y_true, y_pred, plot=False):

    true_to_val = dict(
        pos=1,
        neg=1,
        neu=0,
    )
    y_true_adj = np.array([true_to_val[x] for x in y_true]).astype(float)
    y_pred_adj = y_pred.copy()
    y_pred_adj[y_true == "neg"] = -y_pred_adj[y_true == "neg"]

    precision, recall, _ = precision_recall_curve(y_true_adj, y_pred_adj)
    score = auc(recall, precision)

    if plot:

        p_pos = y_true_adj.sum() / len(y_true_adj)
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="data"))

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[p_pos, p_pos],
                mode="lines",
                name="random",
                line=dict(color="red", dash="solid"),
            )
        )

        fig.update_layout(
            title="PR",
            xaxis_title="recall",
            yaxis_title="precision",
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,  # Link x-axis to y-axis
            ),  # Ensure equal scale
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,  # Link y-axis to x-axis
            ),  # Ensure equal scale
            width=600,
            height=600,
        )

        wandb.log({"pr": fig})

    return score


def compute_acc(data):

    correct_count = 0
    total_count = len(data)

    for concept, values in data.items():
        pos = values.get("pos", 1)
        neg = values.get("neg", -1)
        neu = values.get("neu", 0)

        if pos > 0 and neg < 0 and abs(neu) < abs(pos) and abs(neu) < abs(neg):
            correct_count += 1

    average = correct_count / total_count if total_count > 0 else 0

    return average


def compute_correlation(
    intervention_data, concept_coefs, x_concepts, raw_data, n_concepts=None, debug=True
):
    corr_dict = {}

    n_concept = concept_coefs.shape[0]

    all_corr = 0
    all_p_value = 0

    if n_concepts is None:
        n_concepts = len(concept_coefs.index)

    concept_names = list(intervention_data["On"].keys())

    for concept in concept_names:

        concept_coef = concept_coefs.loc[concept, :]
        C_true = np.exp(concept_coef)

        on_data = intervention_data["On"][concept]
        on_data = np.expm1(on_data)
        on_data = on_data * raw_data.sum(axis=1, keepdims=True) / 1e4

        off_data = intervention_data["Off"][concept]
        off_data = np.expm1(off_data)
        off_data = off_data * raw_data.sum(axis=1, keepdims=True) / 1e4

        estim_coef = on_data.mean(axis=0) / off_data.mean(axis=0)

        corr, p_value = pearsonr(estim_coef, C_true)

        if debug:
            corr_dict[concept + "_corr"] = corr
            corr_dict[concept + "_p_value"] = p_value
        all_corr += corr
        all_p_value += p_value

    corr_dict["avg_concept_coef_corr"] = all_corr / n_concept
    corr_dict["avg_concept_coef_corr_p_value"] = all_p_value / n_concept

    return corr_dict


def score_intervention(
    metrics: List[Literal["auroc", "auprc", "acc", "corr"]],
    scores=None,
    data=None,
    concept_coefs=None,
    concepts=None,
    raw_data=None,
    n_concepts=None,
    debug=False,
    plot=False,
    return_df=False,
):

    metric_funs = dict(
        auroc=compute_auroc,
        auprc=compute_auprc,
        acc=compute_acc,
        corr=compute_correlation,
    )

    flat_list = helpers.flatten_to_list_of_lists(scores)
    df_long = pd.DataFrame(flat_list)
    columns = ["intervention", "concept", "direction", "values"]
    df_long.columns = columns

    y_true = df_long.direction.values
    y_pred = df_long["values"].values

    results = dict()
    for metric in metrics:
        if metric == "acc":
            results["On_acc"] = metric_funs[metric](scores["On"])
            results["Off_acc"] = metric_funs[metric](scores["Off"])
            results["avg_acc"] = (results["On_acc"] + results["Off_acc"]) / 2

        elif metric == "corr":
            corr_results = metric_funs[metric](
                data, concept_coefs, concepts, raw_data, n_concepts, debug
            )
            results.update(corr_results)

        else:
            results[metric] = metric_funs[metric](y_true, y_pred, plot=plot)

    if return_df:
        results, df

    return results


def intervene(
    x_values,
    concepts: pd.DataFrame,
    model: Module,
    on_concepts: List[str] | None = None,
    off_concepts: List[str] | None = None,
):

    on_concepts = [] if on_concepts is None else on_concepts
    off_concepts = [] if off_concepts is None else off_concepts

    concepts_ivn = concepts.values.copy()
    mask = np.zeros_like(concepts)

    np.array([])

    for concept_name in on_concepts:
        column = concepts.columns.get_loc(concept_name)
        mask[:, column] = 1
        concepts_ivn[:, column] = 1

    for concept_name in off_concepts:
        column = concepts.columns.get_loc(concept_name)
        mask[:, column] = 1
        concepts_ivn[:, column] = 0

    x_ivn = model.intervene(
        helpers._to_tensor(x_values),
        helpers._to_tensor(concepts_ivn),
        helpers._to_tensor(mask),
    )["x_pred"]

    x_ivn = x_ivn.detach().numpy()

    x_ivn = pd.DataFrame(
        x_ivn,
        index=x_values.index,
        columns=x_values.columns,
    )

    return x_ivn


def evaluate_intervention_emd_with_target(
    x_train,
    x_ivn,
    x_target,
    labels_train,
    pre_computed_emd_train=None,
) -> Dict[str, Any]:

    if pre_computed_emd_train is None:
        source = np.unique(labels_train)
        scores = np.zeros_like(source)

        for k, s in enumerate(source):
            x_source = x_train[labels_train == s]
            scores[k] = met.emd(x_target, x_source)

        min_train_score = np.min(scores)
    else:
        min_train_score = pre_computed_emd_train

    ivn_score = met.emd(x_target, x_ivn)

    emd_ratio = ivn_score / (min_train_score + 1e-8)

    score = dict(emd_ratio=emd_ratio, pre_computed_emd_train=min_train_score)

    return score


def _cell_level_metric_align(
    adata_ivn_pred,  # Taking the unstimulated data and changing the stimulation concept to 1 and see what it predicts.
    adata_ivn_true,  # Stimulated data with stimulation of interest.
    align_on: str | None = None,
    obsm_key: str = "X",
) -> Dict[str, Any]:

    # check dimensions
    if adata_ivn_pred.shape[0] != adata_ivn_true.shape[0]:
        raise ValueError(
            "adata_ivn_pred and adata_ivn_true must have the same number of samples."
        )

    if adata_ivn_pred.shape[1] != adata_ivn_true.shape[1]:
        raise ValueError(
            "adata_ivn_pred and adata_ivn_true must have the same number of features."
        )

    if obsm_key == "X":
        print("Evaluate on X")
        X_pred = adata_ivn_pred.to_df()
        X_true = adata_ivn_true.to_df()

    else:
        print("Evaluate on obsm key: {}".format(obsm_key))
        X_pred = pd.DataFrame(
            adata_ivn_pred.obsm[obsm_key], index=adata_ivn_pred.obs_names
        )
        X_true = pd.DataFrame(
            adata_ivn_true.obsm[obsm_key], index=adata_ivn_true.obs_names
        )

    if align_on is not None:
        print("Aligning on label: {}".format(align_on))

        X_pred.index = adata_ivn_pred.obs[align_on]
        X_true.index = adata_ivn_true.obs[align_on]

        X_pred = X_pred.loc[X_true.index]

    return X_pred, X_true


def evaluate_intervention_cell_level_mse(
    adata_ivn_pred: ad.AnnData,  # Taking the unstimulated data and changing the stimulation concept to 1 and see what it predicts.
    adata_ivn_true: ad.AnnData,  # Stimulated data with stimulation of interest.
    align_on: str | None = None,
    obsm_key: str = "X",
) -> Dict[str, Any]:

    print("Evaluating on {} cells".format(adata_ivn_pred.shape[0]))

    X_pred, X_true = _cell_level_metric_align(
        adata_ivn_pred,
        adata_ivn_true,
        align_on,
        obsm_key,
    )

    mse = np.mean((X_true.values - X_pred.values) ** 2)

    score = dict(cell_level_mse=mse)

    return score


def evaluate_intervention_mmd_with_target(
    x_train,  # All unstimulated data + and everything stim except your stimulation of interest.
    x_ivn,  # Taking the unstimulated data and changing the stimulation concept to 1 and see what it predicts.
    x_target,  # Stimulated data with stimulation of interest.
    labels_train,  # Cell type + stimulation
    pre_computed_mmd_train=None,
) -> Dict[str, Any]:

    gamma = 1 / x_train.shape[-1]

    if pre_computed_mmd_train is None:
        source = np.unique(labels_train)
        scores = np.zeros_like(source)

        for k, s in enumerate(source):
            x_source = x_train[labels_train == s]
            if x_source.shape[0] < 10:
                scores[k] = np.inf
            else:
                scores[k] = met.calculate_mmd(x_target, x_source, gamma)

        min_train_score = np.min(scores)
    else:
        min_train_score = pre_computed_mmd_train

    ivn_score = met.calculate_mmd(x_target, x_ivn, gamma)

    mmd_ratio = ivn_score / (min_train_score + 1e-8)

    score = dict(mmd_ratio=mmd_ratio, pre_computed_mmd_train=min_train_score)

    return score


def evaluate_intervention_r2_with_target(
    x_og,
    x_new,
    x_target,
    method: Literal["pearson", "mse"] = "pearson",
) -> Dict[str, Any]:
    from sklearn.metrics import r2_score

    mean_og = x_og.mean(axis=0)
    mean_new = x_new.mean(axis=0)
    mean_target = x_target.mean(axis=0)

    new_score = r2_score(mean_target, mean_new)
    old_score = r2_score(mean_target, mean_og)

    score = dict(
        pre_ivn_r2_score=old_score,
        post_ivn_r2_score=new_score,
    )

    return score


def evaluate_intervention_DE_with_target(
    x_train,
    x_ivn,
    x_target,
    genes_list,
) -> Dict[str, Any]:
    """
    Computes the recall for significantly up and down-regulated genes as obtained with the true treated population and the predicted one.

    1. We first compute DE genes between x_train and x_target
    2. We compute DE genes between x_train and x_ivn
    3. We compare the list of DE genes for each direction (upregulated and downregulated)

    Inputs:
        x_train: array of control data - this should already be normalized
        x_ivn: array of predicted intervention data - should be normalized
        x_target: array of target intervention data - should be normalized
        list_of_genes: the list of gene_names for the DE analysis (can be just a list of strings with same lenghts as number of columns in x_train)
    Outputs:
        recall_pos: recall for upregulated genes (# correctly predicted upregulated DE genes / # true upregulated DE genes)
        recall_negs: recall for downregulated genes (# correctly predicted downregulated DE genes / # true downregulated DE genes)
    """

    adata_train = ad.AnnData(
        X=x_train,
        var=pd.DataFrame(index=genes_list),
        obs=pd.DataFrame({"treated": ["FALSE"] * len(x_train)}),
    )
    adata_ivn = ad.AnnData(
        X=x_ivn,
        var=pd.DataFrame(index=genes_list),
        obs=pd.DataFrame({"treated": ["TRUE"] * len(x_ivn)}),
    )
    adata_target = ad.AnnData(
        X=x_target,
        var=pd.DataFrame(index=genes_list),
        obs=pd.DataFrame({"treated": ["TRUE"] * len(x_target)}),
    )

    adata_train.obs_names = [f"train_{x}" for x in adata_train.obs_names]
    adata_target.obs_names = [f"target_{x}" for x in adata_target.obs_names]
    adata_ivn.obs_names = [f"ivn_{x}" for x in adata_ivn.obs_names]

    adata_true = ad.concat([adata_train, adata_target])
    adata_pred = ad.concat([adata_train, adata_ivn])

    adata_true.obs["treated"] = adata_true.obs["treated"].astype("category")
    adata_pred.obs["treated"] = adata_pred.obs["treated"].astype("category")

    # Run DE: A vs B (two-sided Wilcoxon)
    sc.tl.rank_genes_groups(
        adata_true, groupby="treated", method="wilcoxon", use_raw=False
    )
    sc.tl.rank_genes_groups(
        adata_pred, groupby="treated", method="wilcoxon", use_raw=False
    )

    # Export tidy table for group Treated vs Untreated
    def scanpy_de_to_df(adata, group="B"):
        keys = ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]
        cols = {k: adata.uns["rank_genes_groups"][k][group] for k in keys}
        return pd.DataFrame(cols)

    res_true = scanpy_de_to_df(adata_true, group="TRUE")
    res_pred = scanpy_de_to_df(adata_pred, group="TRUE")

    true_pos = res_true.loc[
        (res_true["pvals_adj"] < 0.001) & (res_true["logfoldchanges"] > 0), "names"
    ].values
    true_negs = res_true.loc[
        (res_true["pvals_adj"] < 0.001) & (res_true["logfoldchanges"] < 0), "names"
    ].values

    pred_pos = res_pred.loc[
        (res_pred["pvals_adj"] < 0.05) & (res_pred["logfoldchanges"] > 0), "names"
    ].values
    pred_negs = res_pred.loc[
        (res_pred["pvals_adj"] < 0.05) & (res_pred["logfoldchanges"] < 0), "names"
    ].values

    # recall pos and negative genes
    recall_pos = len(set(true_pos) & set(pred_pos)) / (len(true_pos) + 1e-8)
    recall_negs = len(set(true_negs) & set(pred_negs)) / (len(true_negs) + 1e-8)

    score = dict(recall_pos=recall_pos, recall_negs=recall_negs)

    return score


def evaluate_intervention_knn_with_target(
    x_train,
    x_ivn,
    x_target,
    labels_train,
) -> float:

    total_labels = np.concatenate(
        [labels_train, ["target"] * x_target.shape[0]], axis=0
    )
    x_total = np.concatenate([x_train, x_target], axis=0)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1).fit(
        x_total, total_labels
    )

    predict_ivn = knn.predict(x_ivn)

    score = np.mean(predict_ivn == "target")
    return score


def evaluate_intervention_1nn_acc_with_target(x_ivn, x_target, **kwargs) -> float:

    x_ivn = np.asarray(x_ivn.copy())
    x_target = np.asarray(x_target.copy())

    print("subsampling to match sizes...")

    if x_ivn.shape[0] < x_target.shape[0]:
        x_target = x_target[
            np.random.choice(
                np.arange(x_target.shape[0]), x_ivn.shape[0], replace=False
            )
        ]
    elif x_ivn.shape[0] > x_target.shape[0]:
        x_ivn = x_ivn[
            np.random.choice(
                np.arange(x_ivn.shape[0]), x_target.shape[0], replace=False
            )
        ]

    x_concat = np.concatenate([x_ivn, x_target], axis=0)
    label_concat = np.asarray([0] * x_ivn.shape[0] + [1] * x_target.shape[0])

    print("calculating 1NN graph...")

    knn_graph = sklearn.neighbors.kneighbors_graph(x_concat, n_neighbors=1).tocoo()
    nn_ind = knn_graph.col

    print("calculating 1NN acc...")

    nn_acc = np.mean((label_concat[nn_ind] == label_concat)[: x_ivn.shape[0]])

    return nn_acc


def evaluate_intervention_cosine_with_target(
    x_train,
    x_ivn,
    x_target,
    labels_train,
    pre_computed_mmd_train=None,
) -> float:

    x_train_means = [
        x_train[labels_train == s].mean(0) for s in np.unique(labels_train)
    ]
    x_ivn_mean = x_ivn.mean(0)
    x_target_mean = x_target.mean(0)

    train_target_cosine_sim_max = np.max(
        [met.cosine_sim(x_train_mean, x_target_mean) for x_train_mean in x_train_means]
    )
    ivn_target_cosine_sim = met.cosine_sim(x_ivn_mean, x_target_mean)

    return ivn_target_cosine_sim / train_target_cosine_sim_max
