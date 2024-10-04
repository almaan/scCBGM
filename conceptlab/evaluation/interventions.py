from conceptlab.evaluation._base import EvaluationClass
from collections import OrderedDict
from conceptlab.utils import helpers
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Literal, Tuple, List
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    cohen_kappa_score,
)
import torch
import wandb


class DistributionShift(EvaluationClass):
    """
    A class to evaluate distribution shifts between two datasets by comparing
    their statistical properties and concept distributions.

    Inherits from:
        EvaluationClass: A base class that provides common evaluation utilities.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def _score_fun(cls, X_new, X_old):
        d_values = pd.Series(cohens_d(X_new.values, X_old.values), index=X_new.columns)
        return d_values


def eval_intervention(
    intervention_type,
    concept_idx,
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
):

    mask = np.zeros_like(x_concepts)
    mask[:, concept_idx] = 1

    x_concepts_intervene = x_concepts.copy()

    if intervention_type == "On":
        x_concepts_intervene[:, concept_idx] = 1
        indices = np.where(x_concepts[:, concept_idx] == 0)[0]

    else:
        x_concepts_intervene[:, concept_idx] = 0
        indices = np.where(x_concepts[:, concept_idx] == 1)[0]

    x_pred_withIntervention = model.intervene(
        torch.tensor(x_true),
        torch.tensor(x_concepts_intervene),
        torch.tensor(mask),
    )["x_pred"]

    x_pred_withIntervention = x_pred_withIntervention.detach().numpy()

    genetrated_data_after_intervention = pd.DataFrame(
        x_pred_withIntervention, columns=coefs.columns
    )

    # get subset
    subset_genetrated_data_after_intervention = genetrated_data_after_intervention.iloc[
        indices
    ]
    subset_genetrated_data = genetrated_data.iloc[indices]
    subset_true_data = true_data.iloc[indices]

    results = dict(values=[], data=[], coef_direction=[])

    for data_name, data in zip(
        ["perturbed", "genetrated", "original"],
        [
            subset_genetrated_data_after_intervention,
            subset_genetrated_data,
            subset_true_data,
        ],
    ):
        for direction_name, genes in concept_vars.items():
            ndata = data.loc[:, genes].copy()
            ndata = ndata.mean(axis=1).values
            results["values"] += ndata.tolist()
            results["data"] += len(ndata) * [data_name]
            results["coef_direction"] += len(ndata) * [direction_name]

    results = pd.DataFrame(results)
    # for visualization
    results["data"] = pd.Categorical(
        results["data"], ["perturbed", "genetrated", "original"]
    )

    concepts = pd.DataFrame(
        x_concepts_intervene[:, ix_og_concepts],
        index=original_test_concepts.index,
        columns=original_test_concepts.columns,
    )

    # get subset
    subset_original_test_concepts = original_test_concepts.iloc[indices]
    subset_concepts = concepts.iloc[indices]

    scores = DistributionShift.score(
        X_old=subset_genetrated_data,
        X_new=subset_genetrated_data_after_intervention,
        concepts_old=subset_original_test_concepts,
        concepts_new=subset_concepts,
        concept_coefs=coefs,
        use_neutral=False,
    )

    return results, scores


def cohens_d(x, y):
    """
    Calculate Cohen's d for each column between two NxD arrays.
    Args:
    x: A numpy array of shape (N, D) representing the first group.
    y: A numpy array of shape (N, D) representing the second group.

    Returns:
    A numpy array of Cohen's d values for each column.
    """
    # Means of each column for both groups
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((np.std(x, axis=0, ddof=1) ** 2) + (np.std(y, axis=0, ddof=1) ** 2)) / 2
    )

    # Cohen's d calculation for each column
    d_values = (mean_x - mean_y) / (pooled_std + 1e-8)

    return d_values


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


def score_intervention(
    d, metrics: List[Literal["auroc", "auprc"]], plot=False, return_df=False
):

    metric_funs = dict(auroc=compute_auroc, auprc=compute_auprc)

    flat_list = helpers.flatten_to_list_of_lists(d)
    df_long = pd.DataFrame(flat_list)
    columns = ["intervention", "concept", "direction", "values"]
    df_long.columns = columns

    y_true = df_long.direction.values
    y_pred = df_long["values"].values

    results = dict()
    for metric in metrics:
        results[metric] = metric_funs[metric](y_true, y_pred, plot=plot)

    if return_df:
        results, df

    return results
