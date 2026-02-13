import anndata as ad
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def prepare_tensors(
    adata: ad.AnnData, concept_key: str | None = None, layer: str = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts AnnData layers and concept observations into PyTorch tensors.

    Returns:
        tuple: (data_tensor, concept_tensor)
    """
    # 1. Convert genomic data/features to tensor
    # .values is faster than to_df() if you don't need index names
    data = adata.to_df(layer=layer).values 

    data_tensor = to_tensor(data)

    if concept_key is None:
        return data_tensor, None

    # 2. Convert concept labels to tensor
    if concept_key not in adata.obsm:
        raise KeyError(f"'{concept_key}' not found in adata.obsm")

    concepts = adata.obsm[concept_key]
    concept_tensor = to_tensor(concepts)


    return data_tensor, concept_tensor


def to_tensor(x : pd.DataFrame | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Utility to convert various data formats to PyTorch tensors."""
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, pd.DataFrame):
        return torch.from_numpy(x.values.astype(float)).float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(float)).float()
    else:
        raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")

class OptionalDataset(Dataset):
    def __init__(self, data_tensor, opt_tensor=None):
        self.data_tensor = data_tensor
        self.opt_tensor = opt_tensor

    def __getitem__(self, index):
        x = self.data_tensor[index]
        y = self.opt_tensor[index] if self.opt_tensor is not None else None
        return x, y

    def __len__(self):
        return len(self.data_tensor)
