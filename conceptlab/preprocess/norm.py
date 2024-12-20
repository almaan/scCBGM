import numpy as np
import scanpy as sc
import anndata as ad


def normalize_standard_scanpy(adata: ad.AnnData, target_sum=1):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata
