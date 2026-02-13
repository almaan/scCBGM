from typing import Literal, List, Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import anndata as ad
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from tqdm import tqdm

from conceptlab.models.cb_vae import scCBGM as scCBGMModel
from .base import APIModelBase
from .utils import prepare_tensors, to_tensor, OptionalDataset

class scCBGM(APIModelBase):
    """
    scCBGM API Wrapper for single-cell Concept Bottleneck Generative Modeling.

    This class provides a high-level interface to train, encode, and perform
    interventions on single-cell data using a Concept Bottleneck VAE architecture.
    It handles the transition between AnnData objects and PyTorch tensors.

    Attributes
    ----------
    device : torch.device
        The device (CPU or CUDA) where the model and tensors are allocated.
    history : dict
        A dictionary containing training metrics such as 'loss' and 'lr' per epoch.
    """

    def __init__(
        self,
        has_cbm: bool = True,
        n_unknown: int = 128,
        lr: float = 0.0005,
        beta: float = 0.0001,
        n_layers: int = 2,
        concepts_hp: float = 0.005,
        hidden_dim: int = 128,
        min_bottleneck_size: int = 128,
        latent_dim: int = 64,
        orthogonality_hp: float = 0.05,
        use_cosine_loss: bool = False,
        decoder_type: Literal["skip", "residual"] = "skip",
    ):
        """
        Initialize the scCBGM API wrapper.

        Parameters
        ----------
        has_cbm : bool, default=True
            Whether to utilize the Concept Bottleneck Module.
        n_unknown : int, default=128
            Dimension of the unsupervised (unknown) concept space.
        lr : float, default=0.0005
            Learning rate for the Adam optimizer.
        beta : float, default=0.0001
            Weight for the KL divergence term (VAE regularization).
        n_layers : int, default=2
            Number of hidden layers in the encoder and decoder.
        concepts_hp : float, default=0.005
            Hyperparameter weighting the concept prediction loss.
        hidden_dim : int, default=128
            Width of the hidden layers.
        min_bottleneck_size : int, default=128
            Minimum dimension for the internal bottleneck layers.
        latent_dim : int, default=64
            Dimension of the latent space (z).
        orthogonality_hp : float, default=0.05
            Hyperparameter weighting the orthogonality constraint on concepts.
        use_cosine_loss : bool, default=False
            Whether to use cosine similarity for the concept loss.
        decoder_type : {"skip", "residual"}, default="skip"
            Architecture style of the decoder network.
        """
        super().__init__()
        conf_dict = locals().copy()
        conf_dict.pop("self")
        conf_dict.pop("__class__", None)

        self._config: OmegaConf = OmegaConf.create(conf_dict)
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[scCBGMModel] = None
        self._concept_key: Optional[str] = None
        self.history: Dict[str, List[float]] = {"loss": [], "lr": []}

    @property
    def concepts_key(self) -> Optional[str]:
        """
        Return the key used for concept metadata in AnnData.

        Returns
        -------
        str or None
            The concept key assigned during fit.
        """
        return self._concept_key

    def _run_inference(
        self, loader: DataLoader, mode: Literal["encode", "reconstruct", "intervene"]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Internal engine for batch processing and model execution.

        Parameters
        ----------
        loader : DataLoader
            The PyTorch DataLoader containing the input tensors.
        mode : {"encode", "reconstruct", "intervene"}
            The operational mode determining the forward pass logic.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If 'encode', returns (known_concepts, unknown_concepts).
            Otherwise, returns the reconstructed gene expression matrix.
        """
        self._model.eval()
        results: List[torch.Tensor] = []
        u_results: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                batch_c = (
                    batch[1].to(self.device)
                    if len(batch) > 1 and batch[1] is not None
                    else None
                )
                batch_m = (
                    batch[2].to(self.device)
                    if len(batch) > 2
                    else ((batch_c != 0).float() if batch_c is not None else None)
                )

                if mode == "intervene":
                    out = self._model.intervene(batch_x, mask=batch_m, concepts=batch_c)
                else:
                    enc = self._model.encode(batch_x, mask=batch_m, concepts=batch_c)
                    z_dict = self._model.reparametrize(**enc)
                    cbm_dict = self._model.cbm(**z_dict, concepts=batch_c, **enc)

                    if mode == "encode":
                        results.append(
                            cbm_dict.get(
                                "pred_concepts", torch.zeros(batch_x.size(0), 0)
                            ).cpu()
                        )
                        u_results.append(
                            cbm_dict.get(
                                "unknown_concepts", torch.zeros(batch_x.size(0), 0)
                            ).cpu()
                        )
                        continue

                    out = self._model.decode(cbm_dict)

                recon_batch = (
                    out["x_recon"] if "x_recon" in out else out.get("x_pred", out)
                )
                results.append(recon_batch.cpu())

        if mode == "encode":
            return torch.cat(results).numpy(), torch.cat(u_results).numpy()
        return torch.cat(results).numpy()

    def fit(
        self,
        adata: ad.AnnData,
        concept_key: str,
        n_epochs: int = 256,
        batch_size: int = 512,
        lr_gamma: float = 0.997,
        num_workers: int = 0,
        layer: Optional[str] = None,
        categorical_concept_mask: Optional[List[bool]] = None,
    ) -> None:
        """
        Initialize the model and train it on the provided AnnData object.

        Parameters
        ----------
        adata : ad.AnnData
            The annotated data matrix.
        concept_key : str
            The key in `adata.obs` or `adata.obsm` where concept labels are stored.
        n_epochs : int, default=256
            Number of training epochs.
        batch_size : int, default=512
            Size of mini-batches for training.
        lr_gamma : float, default=0.997
            Multiplicative factor for learning rate decay.
        num_workers : int, default=0
            Number of subprocesses to use for data loading.
        layer : str, optional
            The layer in `adata` to use for gene expression. Uses `adata.X` if None.
        categorical_concept_mask : list of bool, optional
            A mask indicating which concepts are categorical. Defaults to all True.
        """
        self._concept_key = concept_key
        data, concepts = prepare_tensors(adata, concept_key, layer)

        conf = OmegaConf.to_container(self._config)
        conf.update(
            {
                "input_dim": data.shape[1],
                "n_concepts": concepts.shape[1],
                "sigmoid_mask": categorical_concept_mask or [True] * concepts.shape[1],
            }
        )

        self._model = scCBGMModel(OmegaConf.create(conf)).to(self.device)
        self._train(data, concepts, n_epochs, batch_size, lr_gamma, num_workers)

    def _train(
        self,
        data: torch.Tensor,
        concepts: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        lr_gamma: float,
        num_workers: int,
    ) -> None:
        """
        Execute the training loop.

        Parameters
        ----------
        data : torch.Tensor
            Gene expression tensor.
        concepts : torch.Tensor
            Concept label tensor.
        num_epochs : int
            Number of epochs to train.
        batch_size : int
            Size of mini-batches.
        lr_gamma : float
            Learning rate decay factor.
        num_workers : int
            Number of workers for the DataLoader.
        """
        torch.set_flush_denormal(True)
        loader = DataLoader(
            TensorDataset(data, concepts),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        self.history = {"loss": [], "lr": []}
        pbar = tqdm(range(num_epochs), desc="Training scCBGM")

        for _ in pbar:
            self._model.train()
            epoch_loss = 0.0
            for x, c in loader:
                x, c = x.to(self.device), c.to(self.device)
                optimizer.zero_grad()
                loss_dict = self._model.loss_function(x, c, **self._model.forward(x))
                loss = loss_dict["Total_loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.history["loss"].append(avg_loss)
            self.history["lr"].append(scheduler.get_last_lr()[0])
            pbar.set_postfix(
                {"loss": f"{avg_loss:.4e}", "lr": f"{self.history['lr'][-1]:.2e}"}
            )
            scheduler.step()

    def encode(
        self,
        adata: ad.AnnData,
        batch_size: int = 128,
        predict_concepts: bool = False,
        concept_key: Optional[str] = None,
        inplace: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Encode the input data into known and unknown concept representations.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data to encode.
        batch_size : int, default=128
            Batch size for inference.
        predict_concepts : bool, default=False
            If True, concepts are predicted from the model rather than read from adata.
        concept_key : str, optional
            Specific key to use for concepts if different from the training key.
        inplace : bool, default=False
            If True, saves 'known_concepts' and 'unknown_concepts' to `adata.obsm`.

        Returns
        -------
        tuple of np.ndarray or None
            (known_concepts, unknown_concepts) if `inplace` is False, else None.
        """
        key = None if predict_concepts else (concept_key or self.concepts_key)
        data, concepts = prepare_tensors(adata, key)
        loader = DataLoader(OptionalDataset(data, concepts), batch_size=batch_size)

        final_k, final_u = self._run_inference(loader, "encode")
        if inplace:
            adata.obsm["known_concepts"], adata.obsm["unknown_concepts"] = (
                final_k,
                final_u,
            )
            return None
        return final_k, final_u

    def reconstruct(
        self,
        adata: ad.AnnData,
        batch_size: int = 128,
        predict_concepts: bool = True,
        concept_key: Optional[str] = None,
        inplace: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Reconstruct the input data using the model's bottleneck.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data to reconstruct.
        batch_size : int, default=128
            Batch size for inference.
        predict_concepts : bool, default=True
            Whether to use predicted concepts for reconstruction.
        concept_key : str, optional
            Specific key for ground-truth concepts if `predict_concepts` is False.
        inplace : bool, default=False
            If True, saves the reconstruction to `adata.layers["reconstruction"]`.

        Returns
        -------
        np.ndarray or None
            The reconstructed data matrix if `inplace` is False, else None.
        """
        key = None if predict_concepts else (concept_key or self.concepts_key)
        data, concepts = prepare_tensors(adata, key)
        loader = DataLoader(OptionalDataset(data, concepts), batch_size=batch_size)

        recon = self._run_inference(loader, "reconstruct")
        if inplace:
            adata.layers["reconstruction"] = recon
            return None
        return recon

    def intervene(
        self,
        adata: ad.AnnData,
        new_concepts: Union[np.ndarray, torch.Tensor],
        concept_key: Optional[str] = None,
        predict_concepts: bool = False,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Perform a concept-based intervention to generate counterfactual reconstructions.

        Parameters
        ----------
        adata : ad.AnnData
            Original data providing the gene expression background.
        new_concepts : np.ndarray or torch.Tensor
            The target concept values to impose on the model.
        concept_key : str, optional
            Key for the baseline concepts to calculate the intervention mask.
        predict_concepts : bool, default=False
            Whether to derive masks based on predicted instead of stored concepts.
        batch_size : int, default=128
            Batch size for inference.

        Returns
        -------
        np.ndarray
            The reconstructed data matrix reflecting the interventions.
        """
        key = None if predict_concepts else (concept_key or self.concepts_key)
        data, old_concepts = prepare_tensors(adata, key)
        new_concepts_t = to_tensor(new_concepts)
        mask = (new_concepts_t != old_concepts).float()

        loader = DataLoader(
            TensorDataset(data, new_concepts_t, mask), batch_size=batch_size
        )
        return self._run_inference(loader, "intervene")

    def decode(
        self,
        known_concepts: np.ndarray,
        unknown_concepts: np.ndarray,
        batch_size: int = 128,
    ) -> np.ndarray:
        """
        Directly map arbitrary concept vectors to the gene expression space.

        Parameters
        ----------
        known_concepts : np.ndarray
            Matrix of supervised concept values.
        unknown_concepts : np.ndarray
            Matrix of latent (unsupervised) concept values.
        batch_size : int, default=128
            Batch size for decoding.

        Returns
        -------
        np.ndarray
            Decoded gene expression matrix.
        """
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(known_concepts).float(),
                torch.from_numpy(unknown_concepts).float(),
            ),
            batch_size=batch_size,
        )

        self._model.eval()
        reconstructions: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_c, batch_u in loader:
                out = self._model.decode(
                    batch_c.to(self.device), batch_u.to(self.device)
                )
                recon_batch = out.get("x_pred", out) if isinstance(out, dict) else out
                reconstructions.append(recon_batch.cpu())
        return torch.cat(reconstructions).numpy()
