import xarray as xr
import anndata as ad
from conceptlab.utils.constants import DimNames, DataVars
from conceptlab.utils.types import NonNegativeFloat
from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import random
import string
import re
import datetime
import csv


def write_dict_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Concept', 'Pos', 'Neg', 'Neu'])
        # Write the rows
        for concept, values in data.items():
            writer.writerow([concept, float(values['pos']), float(values['neg']), float(values['neu'])])


def add_extras_to_concepts(dataset,cfg) -> xr.Dataset:


   # Step 1: Create the new concept coordinate
    new_concepts = dataset['concept'].values.tolist() + \
                   [f'concept_{i}' for i in range( cfg.dataset.n_concepts, cfg.dataset.n_concepts+cfg.dataset.n_tissues)] + \
                   [f'concept_{i}' for i in range( cfg.dataset.n_concepts+cfg.dataset.n_tissues,\
                                                     cfg.dataset.n_concepts+cfg.dataset.n_tissues+cfg.dataset.n_celltypes)]+\
                   [f'concept_{i}' for i in range(cfg.dataset.n_concepts+cfg.dataset.n_tissues+cfg.dataset.n_celltypes,\
                                                    cfg.dataset.n_concepts+cfg.dataset.n_tissues+cfg.dataset.n_celltypes+cfg.dataset.n_batches)] 


    # Step 2: Extend the concepts data variable
    new_concepts_data = np.zeros((dataset.sizes['obs'], len(new_concepts)))

    # Copy the existing concepts values
    new_concepts_data[:, :cfg.dataset.n_concepts] = dataset['concepts'].values

    total_concepts= cfg.dataset.n_concepts
    # Set the appropriate concept_tissue, concept_batch, and concept_celltype coefficients
    for i in range(dataset.sizes['obs']):
        tissue = dataset['tissues'].isel(obs=i).values.item()  # Convert to a scalar
        tissue_index = int(tissue.split('_')[-1])
        new_concepts_data[i, cfg.dataset.n_concepts + tissue_index] = 1
        

        celltype = dataset['celltypes'].isel(obs=i).values.item()  # Convert to a scalar
        celltype_index = int(celltype.split('_')[-1])
        new_concepts_data[i, cfg.dataset.n_concepts +cfg.dataset.n_tissues + celltype_index] = 1
        total_concepts+=cfg.dataset.n_celltypes


        batch = dataset['batches'].isel(obs=i).values.item()  # Convert to a scalar
        batch_index = int(batch.split('_')[-1])
        new_concepts_data[i, cfg.dataset.n_concepts +cfg.dataset.n_tissues+ cfg.dataset.n_celltypes+ batch_index] = 1



    # Step 3: Extend the concept_coef data variable
    new_concept_coef = np.zeros((len(new_concepts), dataset.sizes['var']))
    new_concept_coef[:8, :] = dataset['concept_coef'].values

    # Copy the tissue_coef, batch_coef, and celltype_coef values to the new concept entries
    for i in range(cfg.dataset.n_tissues):
        new_concept_coef[ cfg.dataset.n_concepts  + i, :] = dataset['tissue_coef'].sel(tissue=f'tissue_{i}').values

    for i in range(cfg.dataset.n_celltypes):
        new_concept_coef[cfg.dataset.n_concepts +cfg.dataset.n_tissues   + i, :] = dataset['celltype_coef'].sel(celltype=f'celltype_{i}').values

    for i in range(cfg.dataset.n_batches):
        new_concept_coef[cfg.dataset.n_concepts +cfg.dataset.n_tissues+ cfg.dataset.n_celltypes + i, :] = dataset['batch_coef'].sel(batch=f'batch_{i}').values


    new_dataset = xr.Dataset(
        {
            'data': dataset['data'],
            'concepts': (['obs', 'concept'], new_concepts_data),
            'concept_coef': (['concept', 'var'], new_concept_coef),
            'tissues': dataset['tissues'],
            'tissue_coef': dataset['tissue_coef'],
            'batches': dataset['batches'],
            'batch_coef': dataset['batch_coef'],
            'celltypes': dataset['celltypes'],
            'celltype_coef': dataset['celltype_coef'],
            'std_libsize': dataset['std_libsize'],
            'p_batch': dataset['p_batch'],
            'p_tissue': dataset['p_tissue'],
            'p_celltype_in_tissue': dataset['p_celltype_in_tissue'],
            'p_concept_in_celltype': dataset['p_concept_in_celltype'],
            'baseline': dataset['baseline']
        },
        coords={
            'obs': dataset['obs'],
            'var': dataset['var'],
            'concept': new_concepts,
            'tissue': dataset['tissue'],
            'celltype': dataset['celltype'],
            'batch': dataset['batch']
        }
    )


    return new_dataset

def dataset_to_anndata(
    dataset: xr.Dataset,
    concepts: np.ndarray | None = None,
    adata_path: str | None = None,
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
        adata.obsm["concepts"] = dataset[DataVars.concept.value].values
    else:
        adata.obsm["concepts"] = concepts

    if adata_path is not None:
        adata.write_h5ad(adata_path)
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
