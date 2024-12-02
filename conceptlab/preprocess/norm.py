import numpy as np
import scanpy as sc
import anndata as ad


def normalize_standard_scanpy(adata: ad.AnnData, target_sum=1e4):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata
