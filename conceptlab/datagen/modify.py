import xarray as xr
from conceptlab.utils.types import *
import conceptlab.utils.constants as C
import numpy as np
from typing import List, Literal


def _get_concepts(dataset: xr.Dataset) -> np.ndarray:
    concepts = dataset[C.DataVars.concept.value].to_numpy()
    n_concepts = concepts.shape[1]
    return concepts, n_concepts


def drop_concepts(
    dataset: xr.Dataset,
    n_drop: PositiveInt | PositiveFloat,
) -> np.ndarray:

    concepts, n_concepts = _get_concepts(dataset)

    if n_drop == 0:
        return concepts

    elif n_drop < 1:
        n_drop = int(np.ceil(n_concepts * n_drop))
    elif n_drop > n_concepts:
        ValueError("Can't drop more concepts than number of concepts in data")

    drop_idx = np.random.choice(n_concepts, replace=False, size=n_drop)

    new_concepts = concepts.copy()
    new_concepts[:, drop_idx] = 0

    return new_concepts


def add_concepts(
    dataset: xr.Dataset,
    n_add: PositiveInt | PositiveFloat,
    p_active: float | List[float] = 0.5,
):

    concepts, n_concepts = _get_concepts(dataset)
    n_obs = concepts.shape[0]

    if n_add == 0:
        return concepts
    elif n_add < 1:
        n_add = int(np.ceil(n_concepts * n_add))

    if isinstance(p_active, float):
        p_active = np.array([p_active] * n_add)
    elif len(p_active) != n_add:
        p_active = np.random.choice(p_active, replace=True, size=n_add)

    new_concepts = np.hstack(
        [np.random.binomial(1, p=p, size=n_obs)[:, None] for p in p_active]
    )

    new_concepts = np.hstack((concepts, new_concepts))

    return new_concepts


def add_noise(
    dataset: xr.Dataset,
    p_noise: float = 0.5,
    n_modified: PositiveInt | PositiveFloat | Literal["all"] = "all",
):

    concepts, n_concepts = _get_concepts(dataset)
    n_obs = concepts.shape[0]

    if n_modified == "all":
        n_modified = n_concepts
    elif n_modified == 0:
        return concepts
    elif n_modified < 1:
        n_modified = int(np.ceil(n_concepts * n_modified))
    elif n_modify > n_concepts:
        ValueError("Can't modify more concepts than number of concepts in data")

    mod_idx = np.random.choice(n_concepts, replace=False, size=n_modified)


    noisy_concepts = concepts.copy()
    mask = np.random.binomial(1,p=p_noise, size=int(n_modified * n_obs)).astype(bool)
    mod_concepts = concepts[:, mod_idx].flatten().copy()
    mod_concepts[mask] = 1 - mod_concepts[mask]
    noisy_concepts[:, mod_idx] = mod_concepts.reshape((n_obs, n_modified))

    return noisy_concepts
