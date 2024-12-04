import anndata as ad
import pandas as pd
import numpy as np


def binarize_concepts(
    concepts: pd.DataFrame,
    threshold: float,
):

    Cn = concepts.copy()
    ix = Cn < threshold
    Cn[ix] = 0
    Cn[~ix] = 1
    return Cn


def annotate_set_concepts(
    data: ad.AnnData | pd.DataFrame,
    set_matrix: pd.DataFrame,
    layer: str | None = None,
    inplace: bool = False,
    unit_interval: bool = False,
    temp: float = 1.0,
) -> pd.DataFrame:

    # check data type and cast into pd.DataFrame representation of data
    if isinstance(data, ad.AnnData):
        X = data.to_df(layer=layer)
    elif isinstance(data, pd.DataFrame):
        X = data
    else:
        raise ValueError("data must be eiher ad.AnnData or pd.DataFrame object")

    M = set_matrix

    # align data and pathways
    inter = X.columns.intersection(M.index)
    M = M.loc[inter, :]
    X = X.loc[:, inter]

    # get variance across cells
    Xv = X.values.var(axis=1, keepdims=True)

    # compute difference from mean squared
    Xsq = (X - X.values.mean(axis=1, keepdims=True)) ** 2

    # get sum of squared differnces for each concept
    C = Xsq.dot(M.astype(float))

    # get pathway sizes
    Ms = M.sum(axis=0)

    # normalize w.r.t. pathway size
    C = C / Ms / Xv

    if unit_interval:
        # C = np.tanh(C / temp)
        C = 1 - np.exp(-temp * C)

    return C
