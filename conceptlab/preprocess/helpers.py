import pandas as pd
import numpy as np
from conceptlab.utils.types import *
from typing import Literal, List, Tuple


def _read_reactome_pathway(
    filename: str,
    organism: Literal["human", "mouse"] | None = None,
    **kwargs,
):

    pwy = pd.read_table(filename, header=None)
    pwy = pwy.iloc[:, [0, 1, 3]]
    pwy.columns = ["feature_name", "pathway", "description"]

    filter_dict = dict(
        human="ENSG",
        mouse="ENMUSG",
    )

    if organism is not None and organism in filter_dict:
        filtr = filter_dict[organism]
        keep = [x.startswith(filtr) for x in pwy["feature_name"].values]
        pwy = pwy.iloc[keep, :]

    return pwy


def _read_go_pathway(
    filename: str,
    **kwargs,
):

    pwy = pd.read_csv(
        filename, header=None, index_col=None, skiprows=list(range(0, 38)), sep="\t"
    )

    pwy = pwy.iloc[:, [2, 3, 4, 8]]
    pwy.columns = ["feature_name", "type", "pathway", "database"]
    pwy = pwy.iloc[pwy.database.values == "P"]
    pwy = pwy.iloc[pwy["type"].values == "involved_in"]

    return pwy


def read_pathway(filename: str, source: Literal["reactome", "go"], *args, **kwargs):

    custom_read_funs = dict(
        go=_read_go_pathway,
        reactome=_read_reactome_pathway,
    )

    if source.lower() in custom_read_funs:
        read_fun = custom_read_funs[source.lower()]

    else:
        supported_methods = ", ".join(list(custom_read_funs.keys()))
        raise ValueError("Supported sources: {}".format(supported_methods))

    return read_fun(filename, *args, **kwargs)


def long_format_pathways_to_matrix(
    pathways: pd.DataFrame | str,
    min_gene_cutoff: PositiveInt = 25,
    pathway_key: str = "pathway",
    feature_name_key: str = "feature_name",
) -> pd.DataFrame:

    feature_names = pathways[feature_name_key].values
    uni_features = np.unique(feature_names)
    feature_map = {f: i for i, f in enumerate(uni_features)}
    n_features = len(feature_map)

    pathways["_ix"] = pathways[feature_name_key].map(feature_map)

    grouped = pathways.groupby(pathway_key)["_ix"].apply(list)
    n_pathways = len(grouped)

    M = np.zeros((n_pathways, n_features))
    for ii in range(n_pathways):
        M[ii, grouped.values[ii]] = 1
    M = pd.DataFrame(M, columns=uni_features, index=grouped.index).T

    return M


def gene_name_conversion(
    source: List[str],
    lookup_table: pd.DataFrame | str,
    source_col: str = "gene_name",
    target_col: str = "ensg",
    drop_isoform: bool = True,
) -> Tuple[List[str], List[str]]:

    if isinstance(lookup_table, str):
        lt = pd.read_csv(lookup_table, header=0, index_col=0)
    else:
        lt = lookup_table.copy()

    lt.index = lt[source_col]

    lt = lt[~lt.index.duplicated(keep="first")]
    if drop_isoform:
        lt[target_col] = [x.split(".")[0] for x in lt[target_col].values]

    result = lt.reindex(source)[target_col].fillna(pd.NA)

    return result
