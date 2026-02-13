import anndata as ad
import scanpy as sc

def default_normalization(adata: ad.AnnData, target_sum=1):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata
