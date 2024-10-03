import anndata as ad
import scanpy as sc
from typing import Dict
from itertools import combinations
import numpy as np
import pandas as pd
from . _lisi import compute_lisi
from . _modularity import compute_modularity
from scipy.spatial import cKDTree as KDTree


def lisi(
    adatas: Dict[str, ad.AnnData] | ad.AnnData, label: str | None = None, **kwargs
):
    from sklearn.decomposition import PCA

    label = "_ident" if label is None else label

    if isinstance(adatas, ad.AnnData):
        ad_all = adatas
    else:
        ad_all = ad.concat(
            adatas,
            axis=0,
            label=label,
        )

    labels = ad_all.obs[label]
    ad_all.obs[label] = pd.Categorical(labels)
    uni_labels = np.unique(labels).tolist()

    combos = combinations(uni_labels, 2)

    scores = dict()
    for combo in combos:
        l1, l2 = combo
        sel_idx = np.isin(labels, combo)

        metadata = pd.DataFrame(labels[sel_idx], columns = [label])

        X = ad_all[sel_idx].to_df().values
        n_components = min(X.shape[1],50)
        pca = PCA(n_components = n_components)
        X = pca.fit_transform(X)

        score = compute_lisi(X, metadata, label_colnames=[label])

        score = np.mean((score - 1))

        scores[f"{l1}_vs_{l2}"] = score

    return scores


def modularity(
        adatas: Dict[str, ad.AnnData] | ad.AnnData, label: str | None = None, embedd: bool = True, **kwargs
):
    from sklearn.decomposition import PCA

    label = "_ident" if label is None else label

    if isinstance(adatas, ad.AnnData):
        ad_all = adatas
    else:
        ad_all = ad.concat(
            adatas,
            axis=0,
            label=label,
        )

    labels = ad_all.obs[label]
    ad_all.obs[label] = pd.Categorical(labels)
    uni_labels = np.unique(labels).tolist()

    combos = combinations(uni_labels, 2)

    scores = dict()
    for combo in combos:
        l1, l2 = combo
        sel_idx = np.isin(labels, combo)

        metadata = labels[sel_idx]

        X = ad_all[sel_idx].to_df().values
        if embedd:
            n_components = min(X.shape[1],50)
            pca = PCA(n_components = n_components)
            X = pca.fit_transform(X)

        score = compute_modularity(X, metadata)

        scores[f"{l1}_vs_{l2}"] = 1 - score

    return scores


