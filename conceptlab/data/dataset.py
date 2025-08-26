import scanpy as sc
from conceptlab.data.data_utils import split_data_for_counterfactuals
import anndata as ad
import numpy as np

class InterventionDataset:
    def __init__(self, data_path, intervention_labels, concept_key, label_variable):
        """
        Loads and preprocesses single cell data
        Inputs:
        - data_path: path to the anndata
        - intervention_labels:
            - hold_out_label: label for the hold-out group (the ground truth to compare agaisnt)
            - mod_label: label for the group that we will modulate / intervene on.
        - label_variable: name of the column in the anndata where the mod_label and hold_out_label are defined.
        """
        print("Loading and preprocessing data...")
        
        print(intervention_labels)
        adata = ad.read_h5ad(data_path)
        sc.pp.normalize_total(adata, target_sum=np.median(adata.X.toarray().sum(axis=1)))
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2048, subset=True)


        hold_out_label = intervention_labels.hold_out_label
        mod_label = intervention_labels.mod_label

        adata, adata_train, adata_test, adata_inter = split_data_for_counterfactuals(
            adata, hold_out_label, mod_label, label_variable
        )

        self.adata = adata
        self.adata_train = adata_train
        self.adata_test = adata_test
        self.adata_inter = adata_inter

        self.concept_key = concept_key
        self.hold_out_label = hold_out_label

    def get_anndatas(self):
        return self.adata, self.adata_train, self.adata_test, self.adata_inter
