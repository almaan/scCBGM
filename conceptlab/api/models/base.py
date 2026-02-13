from abc import ABC, abstractmethod
import anndata as ad
import numpy as np
import pandas as pd
import torch

class APIModelBase:
    def __init__(
        self,
    ):

        self._fitted = False
        self._model = None
        self._concept_key = None
        self._history = {}

    @abstractmethod
    def fit(self, adata: ad.AnnData, concept_key: str | None = None):
        pass

    @abstractmethod
    def decode(self, adata: ad.AnnData, concept_key: str | None = None):
        pass

    @abstractmethod
    def encode(self, adata: ad.AnnData, concept_key: str | None = None):
        pass

    @property
    def training_history(
        self,
    ):
        return self._history

    @abstractmethod
    def intervene(
        self,
        adata: ad.AnnData,
        new_concepts: np.ndarray | torch.Tensor | pd.DataFrame,
        concept_key: str | None = None,
    ):
        pass
