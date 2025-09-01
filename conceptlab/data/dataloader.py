import pytorch_lightning as pl
import anndata as ad
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch as t
import numpy as np
import pandas as pd

import xarray as xr


class GeneExpressionDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        add_concepts: bool = False,
        concepts: pd.DataFrame | np.ndarray = None,
    ):

        if isinstance(data, np.ndarray):
            self.data = data
            self.var_names = None
            self.obs_names = None
        elif isinstance(data, pd.DataFrame):
            self.var_names = data.columns.tolist()
            self.obs_names = data.index.tolist()
            self.data = data.values.astype(np.float32)
            self.data = t.tensor(self.data)
        else:
            raise ValueError()

        self.add_concepts = add_concepts
        if self.add_concepts:
            self.concepts = concepts.values.astype(np.float32)
            self.concepts = t.tensor(self.concepts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.add_concepts:
            concepts = self.concepts[idx]
            return sample, concepts
        return sample, t.tensor([])

    def get_data(
        self,
    ):
        return pd.DataFrame(self.data, index=self.obs_names, columns=self.var_names)


class GeneExpressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame | ad.AnnData | xr.Dataset,
        batch_size: int = 32,
        val_split: float = 0.2,
        layer: str | None = None,
        obsm_key: str | None = None, # <-- Added parameter
        add_concepts: bool = False,
        concept_key: str = "concepts",
        normalize: bool = True,
        split_by: str | None = "split_label",
    ):
        super().__init__()

        self.add_concepts = add_concepts
        self.split_by = split_by

        if isinstance(data, xr.Dataset):
            self.data = data.data.to_dataframe().unstack()
            if self.add_concepts:
                self.concepts = data.concepts.to_dataframe().unstack()
        elif isinstance(data, ad.AnnData):
            # --- MODIFICATION START ---
            if obsm_key:
                print(f"Using data from .obsm['{obsm_key}']")
                # Ensure the data from obsm is a DataFrame with the correct index
                self.data = pd.DataFrame(data.obsm[obsm_key], index=data.obs.index)
            else:
                # Fallback to the original behavior if obsm_key is not provided
                print(f"Using data from .X or layer='{layer}'")
                self.data = data.to_df(layer=layer)
            # --- MODIFICATION END ---

            if self.add_concepts:
                self.concepts = pd.DataFrame(data.obsm[concept_key], index=data.obs.index)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError(
                "Only supported data formats are : ad.AnnData and pd.DataFrame"
            )

        if not self.add_concepts:
            self.concepts = None

        if normalize:
            print("Applying log(1+CPM) normalization...")
            X = self.data.values.astype(np.float32)
            # Avoid division by zero for rows with all zeros
            row_sums = np.sum(X, axis=1, keepdims=True)
            safe_row_sums = np.where(row_sums == 0, 1, row_sums)
            X = X / safe_row_sums * 1e4
            X = np.log1p(X)
            self.data = pd.DataFrame(X, index=self.data.index, columns=self.data.columns)

        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):

        dataset = GeneExpressionDataset(self.data, self.add_concepts, self.concepts)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
