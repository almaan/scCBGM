import anndata as ad
import pandas as pd
import numpy as np
from conceptlab.utils.types import *
import conceptlab.utils.constants as C
from typing import Literal

def read_reactome_pathway(filename: str, organsim: Literal['human','mouse'] | None = None):
    pwy = pd.read_table(filename, header = None)
    pwy = pwy.iloc[:,[0,1,3]]
    pwy.columns = ['feature_name','pathway','description']

    filter_dict = dict(human = 'ENSG',
                       mouse = 'ENMUSG',
                       )

    if organism is not None and organism in filter_dict:
        filtr = filter_dict[organism]
        keep = [x for x in pwy.index if x.startswith(filtr)]
        pwy = pwy.iloc[keep,:]

    return pwy


def reactome_pathways_to_matrix(pathways: pd.DataFrame | str,
                               min_gene_cutoff: PositiveInt = 25,
                               pathway_key: str = 'pathway',
                               feature_name_key: str = 'feature_name',
                               )->pd.DataFrame:

    pathways_cnts = pathways[pathway_key].value_counts()
    top_names = np.array(pathways_cnts.index[pathways_cnts > min_gene_cutoff])
    pathways = pathways.iloc[np.isin(pathways[pathway_key].values,top_names)]

    mat = pathways[[pathway_key]]
    mat.index = pathways[feature_name_key]

    # Get unique feature names and pathways
    unique_features = mat.index.unique()
    unique_pathways = mat[pathway_key].unique()

    # Initialize the matrix with zeros
    M = pd.DataFrame(0, index=unique_features, columns=unique_pathways)

    # Populate the matrix
    for feature_name, pathway in mat.iterrows():
        M.loc[feature_name, pathway] = 1

    return M


def read_reactome_pathway(filename: str, organism: Literal['human','mouse'] | None = None):
    pwy = pd.read_table(filename, header = None)
    pwy = pwy.iloc[:,[0,1,3]]
    pwy.columns = ['feature_name','pathway','description']

    filter_dict = dict(human = 'ENSG',
                       mouse = 'ENMUSG',
                       )

    if organism is not None and organism in filter_dict:
        filtr = filter_dict[organism]
        keep = [x for x in pwy.index if x.startswith(filtr)]
        pwy = pwy.iloc[keep,:]

    return pwy




