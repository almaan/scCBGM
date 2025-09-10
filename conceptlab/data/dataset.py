import scanpy as sc
from conceptlab.data.data_utils import split_data_for_counterfactuals
import anndata as ad
import numpy as np
from typing import List, Union 
import logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def normalize_sc_data(adata, target_sum, variable_genes = True):
    #target_sum = np.median(adata.X.sum(axis=1)) if isinstance(adata.X, np.ndarray) else np.median(adata.X.toarray().sum(axis=1))
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if variable_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    return adata

class InterventionDataset:
    def __init__(self, data_path, intervention_labels, concept_key, mmd_label = None, single_cell_preproc = True, target_sum = 1000 ):
        """
        Loads and preprocesses single cell data
        Inputs:
        - data_path: path to the anndata
        - intervention_labels:
            - hold_out_label: label for the hold-out group (the ground truth to compare agaisnt)
            - mod_label: label for the group that we will modulate / intervene on.
        - single_cell_preproc: whether to preprocess the cells using scanpy.
        - mmd_label: label used to compute mmd ratios.
        """
        print("Loading and preprocessing data...")
        print(intervention_labels)

        self.concept_key = concept_key
        self.hold_out_label = intervention_labels.hold_out_label
        self.mod_label = intervention_labels.mod_label
        self.concepts_to_flip = intervention_labels.concepts_to_flip
        self.control_reference = intervention_labels.reference # The value of controls in the concepts to flip

        self.label_variable = intervention_labels.label_variable
        self.mmd_label = mmd_label

        adata = ad.read_h5ad(data_path)

        if single_cell_preproc:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            adata.layers["og"] = adata.X.copy()  # preserve counts (after normalization)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)

        if not isinstance(adata.X, np.ndarray):
            adata.X = adata.X.toarray()

        if not isinstance(self.label_variable,str): # it's a list then
            log.info("Creating a joint label variable")
            adata.obs["_".join(self.label_variable)] = adata.obs[self.label_variable].agg("_".join, axis=1)
            self.label_variable = "_".join(self.label_variable)

        adata, adata_train, adata_test, adata_inter = split_data_for_counterfactuals(
            adata, self.hold_out_label, self.mod_label, self.label_variable
        )

        adata.uns['pc_transform'] = PCA(n_components=128).fit(adata_train.X)

        for x_data in [adata, adata_train, adata_test, adata_inter]:
            x_data.uns['pc_transform'] = adata.uns['pc_transform']
            x_data.obsm['X_pca'] = x_data.uns['pc_transform'].transform(x_data.X)

        self.adata = adata
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.adata_inter = adata_inter

    def get_anndatas(self):
        return self.adata, self.adata_train, self.adata_test, self.adata_inter