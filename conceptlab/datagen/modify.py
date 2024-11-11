import xarray as xr
from conceptlab.utils.types import *
import conceptlab.utils.constants as C
import numpy as np
from typing import List, Literal
import pandas as pd

DTYPE_STR = "<U64"


def _get_concepts(
    dataset: xr.Dataset | None, concepts: pd.DataFrame | None = None, return_coefs=False
) -> np.ndarray:
    if dataset is not None:
        _concepts = dataset[C.DataVars.concept.value].values
        concepts = pd.DataFrame(
            _concepts,
            index=dataset.coords["obs"].values,
            columns=dataset.coords["concept"].values,
        )

    elif concepts is None:
        raise ValueError("Ond of dataset and concepts has to not be None")
    n_concepts = concepts.shape[1]

    coefs = dataset["concept_coef"].to_dataframe().unstack()["concept_coef"].T.copy()

    if return_coefs:
        return concepts, n_concepts, coefs

    return concepts, n_concepts


def _get_num(n_og: PositiveInt, n_sel: PositiveFloat):
    if n_sel < 1:
        return int(np.ceil(n_og * n_sel))
    return n_sel


def _get_indicator(n_concepts: PositiveInt) -> np.ndarray:
    return np.array([C.Mods.none] * n_concepts, dtype=DTYPE_STR)


def drop_concepts(
    dataset: xr.Dataset | None = None,
    concepts: pd.DataFrame | None = None,
    n_drop: PositiveInt | PositiveFloat = 1,
    **kwargs,
) -> np.ndarray:

    concepts, n_concepts = _get_concepts(dataset, concepts)
    indicator = _get_indicator(n_concepts)

    if n_drop == 0:
        return concepts, indicator

    elif n_drop < 1:
        n_drop = int(np.ceil(n_concepts * n_drop))
    elif n_drop > n_concepts:
        ValueError("Can't drop more concepts than number of concepts in data")

    drop_idx = np.random.choice(n_concepts, replace=False, size=n_drop)

    concepts = concepts.drop(columns=concepts.columns[drop_idx])

    indicator = pd.Series(indicator, index=concepts.columns)

    return concepts, indicator


def add_concepts(
    dataset: xr.Dataset | None = None,
    concepts: np.ndarray | None = None,
    n_add: PositiveInt | PositiveFloat = 1,
    p_active: float | List[float] = 0.5,
    **kwargs,
) -> np.ndarray:

    concepts, n_concepts = _get_concepts(dataset, concepts)
    n_obs = concepts.shape[0]
    indicator = _get_indicator(n_concepts)

    if n_add == 0:
        return concepts, indicator
    elif n_add < 1:
        n_add = int(np.ceil(n_concepts * n_add))

    if isinstance(p_active, float):
        p_active = np.array([p_active] * n_add)
    elif len(p_active) != n_add:
        p_active = np.random.choice(p_active, replace=True, size=n_add)

    new_concepts = np.hstack(
        [np.random.binomial(1, p=p, size=n_obs)[:, None] for p in p_active]
    )

    new_concepts = pd.DataFrame(
        new_concepts,
        index=concepts.index,
        columns=[f"{C.Mods.add}_{k}" for k in range(n_add)],
    )

    new_concepts = pd.concat((concepts, new_concepts), axis=1)

    indicator = np.append(indicator, np.array([C.Mods.add] * n_add, dtype=DTYPE_STR))
    indicator = pd.Series(indicator, index=new_concepts.columns)

    return new_concepts, indicator


def add_noise(
    dataset: xr.Dataset | None = None,
    concepts: np.ndarray | None = None,
    p_noise: float = 0.5,
    n_modified: PositiveInt | PositiveFloat | Literal["all"] = "all",
    **kwargs,
) -> np.ndarray:

    concepts, n_concepts = _get_concepts(dataset, concepts)
    n_obs = concepts.shape[0]

    indicator = _get_indicator(n_concepts)

    if n_modified == "all":
        n_modified = n_concepts
    elif n_modified == 0:
        return concepts, indicator
    elif n_modified < 1:
        n_modified = int(np.ceil(n_concepts * n_modified))
    elif n_modified > n_concepts:
        ValueError("Can't modify more concepts than number of concepts in data")

    mod_idx = np.random.choice(n_concepts, replace=False, size=n_modified)

    mask = np.random.binomial(1, p=p_noise, size=int(n_modified * n_obs)).astype(bool)
    mod_concepts = concepts.values[:, mod_idx].flatten().copy()
    mod_concepts[mask] = 1 - mod_concepts[mask]

    concepts.iloc[:, mod_idx] = mod_concepts.reshape((n_obs, n_modified))

    indicator[mod_idx] = C.Mods.noise
    indicator = pd.Series(indicator, index=concepts.columns)

    return concepts, indicator


def add_duplicate(
    dataset: xr.Dataset | None = None,
    concepts: np.ndarray | None = None,
    n_duplicate: PositiveInt = 1,
    n_replica: PositiveInt = 1,
    **kwargs,
):

    concepts, n_concepts = _get_concepts(dataset, concepts)
    concepts.shape[0]

    indicator = _get_indicator(n_concepts)
    if n_duplicate == 0:
        return concepts

    _get_num(n_concepts, n_duplicate)

    dup_idx = np.random.choice(n_concepts, replace=False, size=n_duplicate)

    indicator[dup_idx] = np.array(
        [f"{C.Mods.duplicate}_{k}" for k in dup_idx], dtype=DTYPE_STR
    )

    if n_replica > 1:
        dup_idx = np.repeat(dup_idx, n_replica)

    new_concepts = concepts.iloc[:, dup_idx]
    new_concepts.columns = [f"{C.Mods.duplicate}_{k}" for k in dup_idx]

    new_concepts = pd.concat((concepts, new_concepts), axis=1)

    add_indicator = np.array(
        [f"{C.Mods.duplicate}_{k}" for k in dup_idx], dtype=DTYPE_STR
    )
    indicator = np.append(indicator, add_indicator)
    indicator = pd.Series(indicator, index=concepts.columns)

    return new_concepts, indicator
