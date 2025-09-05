import scanpy as sc
from conceptlab.data.data_utils import split_data_for_counterfactuals
import anndata as ad
import numpy as np
from typing import List, Union 
import logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class InterventionDataset:
    def __init__(self, data_path, intervention_labels, concept_key, label_variable: Union[List,str], single_cell_preproc = True, log1p = True):
        """
        Loads and preprocesses single cell data
        Inputs:
        - data_path: path to the anndata
        - intervention_labels:
            - hold_out_label: label for the hold-out group (the ground truth to compare agaisnt)
            - mod_label: label for the group that we will modulate / intervene on.
        - label_variable: name of the column in the anndata where the mod_label and hold_out_label are defined. If label_variable is a list, then we merge the columns 
            to create a new label variable where each unique combination of the columns is a new label. Example if label_variable = ['treatment', 'cell_type'], we'll create a new column 'treatment_cell_type' where each element will be joined with '_'.
        - single_cell_preproc: whether to preprocess the cells using scanpy.
        """
        print("Loading and preprocessing data...")
        
        print(intervention_labels)
        adata = ad.read_h5ad(data_path)

        if single_cell_preproc:
            sc.pp.normalize_total(adata, target_sum=np.median(adata.X.toarray().sum(axis=1)))
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
        
        adata.X = adata.X.toarray()

        if not isinstance(label_variable,str): # it's a list then
            log.info("Creating a joint label variable")
            adata.obs["_".join(label_variable)] = adata.obs[label_variable].agg("_".join, axis=1)
            label_variable = "_".join(label_variable)

        hold_out_label = intervention_labels.hold_out_label
        mod_label = intervention_labels.mod_label

        adata, adata_train, adata_test, adata_inter = split_data_for_counterfactuals(
            adata, hold_out_label, mod_label, label_variable
        )

        adata.uns['pc_transform'] = PCA(n_components=128).fit(adata_train.X)

        for x_data in [adata, adata_train, adata_test, adata_inter]:
            x_data.uns['pc_transform'] = adata.uns['pc_transform']
            x_data.obsm['X_pca'] = x_data.uns['pc_transform'].transform(x_data.X)

        self.adata = adata
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.adata_inter = adata_inter

        self.concept_key = concept_key
        self.hold_out_label = hold_out_label
        self.concepts_to_flip = intervention_labels.concepts_to_flip
        self.control_reference = intervention_labels.reference # The value of controls in the concepts to flip

        self.label_variable = label_variable

    def get_anndatas(self):
        return self.adata, self.adata_train, self.adata_test, self.adata_inter
