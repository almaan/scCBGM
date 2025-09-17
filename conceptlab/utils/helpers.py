import xarray as xr
import anndata as ad
from conceptlab.utils.constants import DimNames, DataVars
from conceptlab.utils.types import NonNegativeFloat
from typing import Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import torch
import random
import string
import re
import datetime
from scipy.sparse import spmatrix
import csv


def write_dict_to_csv(data, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Concept", "Pos", "Neg", "Neu"])
        # Write the rows
        for concept, values in data.items():
            writer.writerow(
                [
                    concept,
                    float(values["pos"]),
                    float(values["neg"]),
                    float(values["neu"]),
                ]
            )


def add_extras_to_concepts(dataset, cfg) -> xr.Dataset:

    # Step 1: Create the new concept coordinate
    new_concepts = (
        dataset["concept"].values.tolist()
        + [
            f"concept_{i}"
            for i in range(
                cfg.dataset.n_concepts, cfg.dataset.n_concepts + cfg.dataset.n_tissues
            )
        ]
        + [
            f"concept_{i}"
            for i in range(
                cfg.dataset.n_concepts + cfg.dataset.n_tissues,
                cfg.dataset.n_concepts
                + cfg.dataset.n_tissues
                + cfg.dataset.n_celltypes,
            )
        ]
        + [
            f"concept_{i}"
            for i in range(
                cfg.dataset.n_concepts
                + cfg.dataset.n_tissues
                + cfg.dataset.n_celltypes,
                cfg.dataset.n_concepts
                + cfg.dataset.n_tissues
                + cfg.dataset.n_celltypes
                + cfg.dataset.n_batches,
            )
        ]
    )

    # Step 2: Extend the concepts data variable
    new_concepts_data = np.zeros((dataset.sizes["obs"], len(new_concepts)))

    # Copy the existing concepts values
    new_concepts_data[:, : cfg.dataset.n_concepts] = dataset["concepts"].values

    total_concepts = cfg.dataset.n_concepts
    # Set the appropriate concept_tissue, concept_batch, and concept_celltype coefficients
    for i in range(dataset.sizes["obs"]):
        tissue = dataset["tissues"].isel(obs=i).values.item()  # Convert to a scalar
        tissue_index = int(tissue.split("_")[-1])
        new_concepts_data[i, cfg.dataset.n_concepts + tissue_index] = 1

        celltype = dataset["celltypes"].isel(obs=i).values.item()  # Convert to a scalar
        celltype_index = int(celltype.split("_")[-1])
        new_concepts_data[
            i, cfg.dataset.n_concepts + cfg.dataset.n_tissues + celltype_index
        ] = 1
        total_concepts += cfg.dataset.n_celltypes

        batch = dataset["batches"].isel(obs=i).values.item()  # Convert to a scalar
        batch_index = int(batch.split("_")[-1])
        new_concepts_data[
            i,
            cfg.dataset.n_concepts
            + cfg.dataset.n_tissues
            + cfg.dataset.n_celltypes
            + batch_index,
        ] = 1

    # Step 3: Extend the concept_coef data variable
    new_concept_coef = np.zeros((len(new_concepts), dataset.sizes["var"]))
    new_concept_coef[:8, :] = dataset["concept_coef"].values

    # Copy the tissue_coef, batch_coef, and celltype_coef values to the new concept entries
    for i in range(cfg.dataset.n_tissues):
        new_concept_coef[cfg.dataset.n_concepts + i, :] = (
            dataset["tissue_coef"].sel(tissue=f"tissue_{i}").values
        )

    for i in range(cfg.dataset.n_celltypes):
        new_concept_coef[cfg.dataset.n_concepts + cfg.dataset.n_tissues + i, :] = (
            dataset["celltype_coef"].sel(celltype=f"celltype_{i}").values
        )

    for i in range(cfg.dataset.n_batches):
        new_concept_coef[
            cfg.dataset.n_concepts
            + cfg.dataset.n_tissues
            + cfg.dataset.n_celltypes
            + i,
            :,
        ] = (
            dataset["batch_coef"].sel(batch=f"batch_{i}").values
        )

    new_dataset = xr.Dataset(
        {
            "data": dataset["data"],
            "concepts": (["obs", "concept"], new_concepts_data),
            "concept_coef": (["concept", "var"], new_concept_coef),
            "tissues": dataset["tissues"],
            "tissue_coef": dataset["tissue_coef"],
            "batches": dataset["batches"],
            "batch_coef": dataset["batch_coef"],
            "celltypes": dataset["celltypes"],
            "celltype_coef": dataset["celltype_coef"],
            "std_libsize": dataset["std_libsize"],
            "p_batch": dataset["p_batch"],
            "p_tissue": dataset["p_tissue"],
            "p_celltype_in_tissue": dataset["p_celltype_in_tissue"],
            "p_concept_in_celltype": dataset["p_concept_in_celltype"],
            "baseline": dataset["baseline"],
        },
        coords={
            "obs": dataset["obs"],
            "var": dataset["var"],
            "concept": new_concepts,
            "tissue": dataset["tissue"],
            "celltype": dataset["celltype"],
            "batch": dataset["batch"],
        },
    )

    return new_dataset


def dataset_to_anndata(
    dataset: xr.Dataset,
    concepts: pd.DataFrame | None = None,
    concept_coef: pd.DataFrame | None = None,
    concept_key: str = "concepts",
    concept_coef_key: str = "concept_coef",
) -> ad.AnnData:

    adata = ad.AnnData(
        dataset.data.values,
        var=pd.DataFrame([], index=dataset[DimNames.var.value].values),
        obs=pd.DataFrame([], index=dataset[DimNames.obs.value].values),
    )

    adata.obs["tissue"] = dataset[DataVars.tissue.value].values
    adata.obs["celltype"] = dataset[DataVars.celltype.value].values
    adata.obs["batch"] = dataset[DataVars.batch.value].values

    if concepts is None:
        _concepts = dataset[DataVars.concept.value].values
        adata.obsm[concept_key] = pd.DataFrame(
            _concepts,
            index=dataset.coords["obs"].values,
            columns=dataset.coords["concept"].values,
        )
    else:
        adata.obsm[concept_key] = concepts

    if concept_coef is None:
        adata.varm[concept_coef_key] = (
            dataset["concept_coef"].to_dataframe().unstack()["concept_coef"].T
        )
    else:
        adata.varm[concept_coef_key] = concept_coef

    return adata


def flatten_to_list_of_lists(d, parent_keys=[]):
    """
    Recursively flatten a nested dictionary into a list of lists,
    where each list contains all keys followed by the final value.

    Args:
    - d: The nested dictionary to flatten.
    - parent_keys: List of parent keys leading to the current level (used during recursion).

    Returns:
    - A list of lists, where each sublist contains the keys and the corresponding value.
    """
    result = []
    for key, value in d.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            # Recurse if the value is another dictionary
            result.extend(flatten_to_list_of_lists(value, current_keys))
        else:
            # Append the final value along with all the keys in a list
            result.append(current_keys + [value])

    return result


def create_composite_image(image_folder, output_image):
    """
    Create a composite image with two rows of images.

    Parameters:
    - image_folder (str): The folder containing the images.
    - output_image (str): The filename for the output composite image.
    """

    # Define the image filenames
    turn_off_filenames = [f"_concept_{i}_turnOff.png" for i in range(8)]
    turn_on_filenames = [f"_concept_{i}_turnOn.png" for i in range(8)]

    # Load the images
    turn_off_images = [
        Image.open(image_folder + filename) for filename in turn_off_filenames
    ]
    turn_on_images = [
        Image.open(image_folder + filename) for filename in turn_on_filenames
    ]

    # Assuming all images are of the same size
    width, height = turn_off_images[0].size

    # Create a new image with double the height and the same width for each row
    composite_image = Image.new("RGB", (width * 8, height * 2))

    # Paste the images into the composite image
    for i in range(8):
        composite_image.paste(turn_on_images[i], (i * width, 0))
        composite_image.paste(turn_off_images[i], (i * width, height))

    # Save the composite image
    composite_image.save(output_image)


def simple_adata_train_test_split(
    adata: ad.AnnData, p_test: NonNegativeFloat = 0.5
) -> Tuple[ad.AnnData, ad.AnnData]:

    if (p_test >= 1) or (p_test <= 0):
        raise ValueError(
            "p_test = {}, this is not in the interval (0,1)".format(p_test)
        )

    idx = np.arange(len(adata))
    np.random.shuffle(idx)
    n_test = int(0.5 * len(adata))

    adata_test, adata_train = adata[idx[0:n_test]].copy(), adata[idx[n_test::]].copy()
    return adata_test, adata_train, n_test, idx


def stratified_adata_train_test_split(
    adata: ad.AnnData,
    p_test: NonNegativeFloat = 0.5,
    concept_key: str = "concepts",
    return_index_only: bool = False,
) -> Tuple[ad.AnnData, ad.AnnData]:
    if concept_key not in adata.obsm:
        raise ValueError(f"{concept_key} not found in `adata.obsm`.")
    if not (0 < p_test < 1):
        raise ValueError(f"p_test = {p_test}, this is not in the interval (0,1)")
    # Create a unique identifier for each combination of concepts
    concept_matrix = np.asarray(adata.obsm[concept_key]).astype(int)
    concept_labels = np.array(["".join(map(str, row)) for row in concept_matrix])

    uni_concept_labels, cnt_concept_labels = np.unique(
        concept_labels, return_counts=True
    )

    drop_labels = np.isin(concept_labels, uni_concept_labels[cnt_concept_labels < 2])

    concept_labels[drop_labels] = uni_concept_labels[np.argmax(cnt_concept_labels)]

    # Perform a stratified split based on the unique concept labels
    indices = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(
        indices, test_size=p_test, stratify=concept_labels, random_state=42
    )

    if return_index_only:
        return dict(train=train_idx, test=test_idx)

    # Split the AnnData object
    adata_test = adata[test_idx].copy()
    adata_train = adata[train_idx].copy()

    # Add split labels for tracking
    adata_test.obs["split_label"] = concept_labels[test_idx]
    adata_train.obs["split_label"] = concept_labels[train_idx]
    return (
        adata_test,
        adata_train,
        len(adata_test),
        np.concatenate((test_idx, train_idx)),
    )


def generate_random_string(length=15):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def clear_cuda_memory():
    # Empty the PyTorch cache
    torch.cuda.empty_cache()
    # Synchronize the GPU
    torch.cuda.synchronize()

    # Optionally, you can also reset memory stats for a fresh start (optional)
    torch.cuda.reset_peak_memory_stats()


def timestamp() -> str:
    return re.sub(":|-|\.| |", "", str(datetime.datetime.now()))


def get_unique_leaf_keys(d):
    unique_keys = set()  # Use a set to collect unique leaf keys

    def recurse_keys(current_dict):
        for key, value in current_dict.items():
            if isinstance(value, dict):  # If the value is a dictionary, recurse into it
                recurse_keys(value)
            elif isinstance(
                value, list
            ):  # If the value is a list, iterate and check if its items are dictionaries
                for item in value:
                    if isinstance(item, dict):
                        recurse_keys(item)
                    else:
                        unique_keys.add(key)  # Add the key if it's a leaf
            else:
                unique_keys.add(key)  # Add the key if it's a leaf

    recurse_keys(d)
    return unique_keys


def get_n_level_keys(d, n):
    n_level_keys = set()

    def recurse_keys(current_dict, current_level):
        # If we have reached the target level (n), add the keys
        if current_level == n:
            n_level_keys.update(current_dict.keys())
        else:
            for key, value in current_dict.items():
                if isinstance(value, dict):  # Only recurse into dictionaries
                    recurse_keys(value, current_level + 1)

    # Start the recursion
    recurse_keys(d, 1)  # Start from level 1
    return n_level_keys


def matrix_correlation(
    O: np.ndarray | spmatrix, P: np.ndarray | spmatrix
) -> np.ndarray:
    # efficient implementation of columnwise
    # correlation between two input matrices
    # shamelessly stolen from: https://github.com/ikizhvatov/efficient-columnwise-correlation/blob/master/columnwise_corrcoef_perf.py

    if isinstance(O, spmatrix):
        O = O.toarray()
    if isinstance(P, spmatrix):
        P = P.toarray()

    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    # compute covariance
    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    # compute variance
    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")

    return cov / np.sqrt(tmp)


def matrix_covariance(O: np.ndarray | spmatrix, P: np.ndarray | spmatrix) -> np.ndarray:
    # efficient implementation of columnwise
    # correlation between two input matrices
    # shamelessly stolen from: https://github.com/ikizhvatov/efficient-columnwise-correlation/blob/master/columnwise_corrcoef_perf.py

    if isinstance(O, spmatrix):
        O = O.toarray()
    if isinstance(P, spmatrix):
        P = P.toarray()

    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    # compute covariance
    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    return cov


def predefined_adata_train_test_split(
    adata: ad.AnnData,
    split_col: str,
    test_label: str,
    ivn_label: str,
    drop_label: str | None = None,
) -> Tuple[ad.AnnData, ...]:
    labels = adata.obs[split_col].values
    if drop_labels is not None:
        drop_idx = labels == drop_label
        adata = adata[~drop_idx].copy()
        labels = adata.obs[split_col].values

    is_test = labels == test_label
    is_ivn == labels == ivn_label

    adata_train = adata[~is_test & ~is_ivn].copy()
    adata_test = adata[is_test].copy()
    adata_ivn = adata[is_ivn].copy()

    return adata_train, adata_ivn, adata_test


def controlled_adata_train_test_split(
    adata: ad.AnnData,
    split_col: str,
    test_labels: str | None = None,
    p_test=0.5,
) -> ad.AnnData:
    from sklearn.model_selection import StratifiedShuffleSplit

    adata_test = adata[test_idx].copy()
    adata_train = adata[train_idx].copy()

    return (
        adata_test,
        adata_train,
        len(adata_test),
        np.concatenate((test_idx, train_idx)),
    )


def custom_adata_train_test_split(
    adata: ad.AnnData,
    split_col: str,
    test_labels: List[str],
    pred_labels: List[str],
    split_pred: bool = False,
    split_pred_p: float | int = 0.2,
    verbose: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:

    labels = adata.obs[split_col].values
    is_test = np.isin(labels, test_labels)
    is_train = ~is_test
    is_pred = np.isin(labels, pred_labels)

    lo_pred_ix = np.array([])
    kp_pred_ix = np.array([])

    if split_pred:

        for label in pred_labels:
            is_label = labels == label
            pred_ix = np.where(is_pred & is_label)[0]
            np.random.shuffle(pred_ix)
            n_pred = len(pred_ix)

            if split_pred_p > 1:
                p_pred = min(n_pred, split_pred_p)
            else:
                p_pred = int(n_pred * split_pred_p)

            lo_pred_ix = np.append(lo_pred_ix, pred_ix[0:p_pred])
            kp_pred_ix = np.append(kp_pred_ix, pred_ix[p_pred::])

        lo_pred_ix = lo_pred_ix.astype(int)
        kp_pred_ix = kp_pred_ix.astype(int)

        is_pred[kp_pred_ix] = False
        is_train[lo_pred_ix] = False

    adata_train = adata[is_train].copy()
    adata_test = adata[is_test].copy()
    adata_pred = adata[is_pred].copy()

    if verbose:
        print("TRAIN")
        print(adata_train.obs[split_col].value_counts())
        print("TEST")
        print(adata_test.obs[split_col].value_counts())
        print("VAL")
        print(adata_pred.obs[split_col].value_counts())

    return adata_train, adata_test, adata_pred


def _to_tensor(x: pd.DataFrame | np.ndarray, device: str = "cpu") -> torch.Tensor:
    if isinstance(x, pd.DataFrame):
        return torch.tensor(x.values.astype(np.float32)).to(device)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x.astype(np.float32)).to(device)
    else:
        raise NotImplementedError


def _to_tensor_gpu(x: pd.DataFrame | np.ndarray) -> torch.Tensor:
    return _to_tensor(x, "cuda:0")


def find_matching_target(on_concepts, off_concepts, target_concepts):

    on_filter = (
        target_concepts[on_concepts].eq(1).all(axis=1)
        if on_concepts
        else pd.Series(True, index=target_concepts.index)
    )
    off_filter = (
        target_concepts[off_concepts].eq(0).all(axis=1)
        if off_concepts
        else pd.Series(True, index=target_concepts.index)
    )

    filtered_observations = target_concepts[on_filter & off_filter]

    if len(filtered_observations) < 1:
        return None

    return filtered_observations


def normalize_counts(x: np.ndarray, total_sum: int = 1e3) -> np.ndarray:
    x = self.data.values.astype(np.float32)
    # Avoid division by zero for rows with all zeros
    row_sums = np.sum(x, axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)
    x = x / safe_row_sums * total_sum
    x = np.log1p(x)
