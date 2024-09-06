import xarray as xr
import anndata as ad
from conceptlab.utils.constants import DimNames, DataVars
from conceptlab.utils.types import NonNegativeFloat
from typing import Tuple
import numpy as np


def dataset_to_anndata(
    dataset: xr.Dataset, adata_path: str | None = None
) -> ad.AnnData:

    adata = ad.AnnData(
        dataset.data.values,
        var=pd.DataFrame([], index=dataset[DimNames.var.value].values),
        obs=pd.DataFrame([], index=dataset[DimNames.obs.value].values),
    )

    adata.obs["tissue"] = dataset[DataVars.tissue.value].values
    adata.obs["celltype"] = dataset[DataVars.celltype.value].values
    adata.obs["batch"] = dataset[DataVars.batch.value].values
    adata.obsm["concepts"] = dataset[DataVars.concept.value].values

    if adata_path is not None:
        adata.write_h5ad(adata_path)
    return adata


def simple_adata_train_test_split(
    adata: ad.AnnData, p_test: NonNegativeFloat = 0.5
) -> Tuple[ad.AnnData, ad.Anndata]:

    if (p_test >= 1) or (p_test <= 0):
        raise ValueError(
            "p_test = {}, this is not in the interval (0,1)".format(p_test)
        )

    idx = np.arange(len(adata))
    np.random.shuffle(idx)
    n_test = int(0.5 * len(adata))

    adata_test, adata_train = adata[idx[0:n_test]].copy(), adata[idx[n_test::]].copy()
    return adata_test, adata_train
