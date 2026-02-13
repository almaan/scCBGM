import numpy as np
import pandas as pd
from typing import List


def curate_concepts(
    adata,
    concept_columns: List[str],
    concept_key: str = "concepts",
    single_column_for_binary: bool = True,
):
    """
    Curates and transforms observation columns into a concept matrix stored in .obsm.
    """
    processed_series = []

    for col in concept_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

        cvals = adata.obs[col]

        # Check if the column is categorical/string
        if pd.api.types.is_object_dtype(cvals) or pd.api.types.is_categorical_dtype(
            cvals
        ):
            n_uniq = cvals.nunique()
            if n_uniq == 2 and single_column_for_binary:
                # Binary: convert to 0/1 using drop_first to get a single column
                dummy = 1 - pd.get_dummies(cvals, drop_first=True, prefix=col).astype(
                    float
                )
                processed_series.append(dummy)
            else:
                # Multi-class: Full One-Hot Encoding
                dummy = pd.get_dummies(cvals, prefix=col)
                processed_series.append(dummy).astype(float)
        else:
            # Numerical: Keep as a Series
            processed_series.append(cvals.to_frame())

    # Concatenate all processed columns into one DataFrame
    concepts_df = pd.concat(processed_series, axis=1)

    # Generate the categorical mask based on the final column count
    # This ensures the mask matches the model's actual concept dimension
    categorical_mask = []
    for col_name in concepts_df.columns:
        # If the column name starts with a prefix from concept_columns and
        # wasn't a raw numerical, it's categorical
        is_cat = any(
            col_name.startswith(f"{orig_col}_") for orig_col in concept_columns
        )
        categorical_mask.append(is_cat)

    # Update AnnData object
    adata.obsm[concept_key] = concepts_df
    adata.uns[f"{concept_key}_categorical_mask"] = categorical_mask

    print(f"Curated {concepts_df.shape[1]} concepts into adata.obsm['{concept_key}']")


def intervene_on_concepts(
    adata,
    concept_key: str = "concepts",
    overrides: dict = None,
    conditions: dict = None,
    inplace: bool = False,
):
    """
    Intervene on specific variables in an existing concept matrix.

    Parameters
    ----------
    adata : ad.AnnData
        The annotated data object.
    concept_key : str, default="concepts"
        The key in adata.obsm containing the concept DataFrame.
    overrides : dict, optional
        Direct mapping of {column_name: new_value}.
        Example: {"cell_type_T-cell": 1.0} forces all cells to be T-cells.
    conditions : dict, optional
        Conditional mapping of {target_column: {if_value: then_value}}.
        Example: {"state": {0.0: 1.0}} changes 'state' to 1.0 only where it was 0.0.
    inplace : bool, default=False
        If True, modifies adata.obsm[concept_key]. If False, returns the modified array.

    Returns
    -------
    np.ndarray or None
        The modified concept matrix if inplace=False.
    """
    if concept_key not in adata.obsm:
        raise KeyError(f"'{concept_key}' not found in adata.obsm")

    # Work on a copy to avoid accidental corruption of original data
    concepts_df = adata.obsm[concept_key].copy()
    if not isinstance(concepts_df, pd.DataFrame):
        # Fallback if it's a numpy array (though curate_concepts stores a DF)
        raise TypeError("concept_key must point to a pandas DataFrame in .obsm")

    # 1. Apply Conditional Interventions
    # Logic: "Where column A is X, set it to Y"
    if conditions:
        for col, mapping in conditions.items():
            if col not in concepts_df.columns:
                print(f"Warning: Column '{col}' not found in concepts. Skipping.")
                continue
            for if_val, then_val in mapping.items():
                concepts_df.loc[concepts_df[col] == if_val, col] = then_val

    # 2. Apply Direct Overrides
    # Logic: "Set all values in column A to X"
    if overrides:
        for col, val in overrides.items():
            if col not in concepts_df.columns:
                print(f"Warning: Column '{col}' not found in concepts. Skipping.")
                continue
            concepts_df[col] = val

    if inplace:
        adata.obsm[concept_key + "_new"] = concepts_df
        return None

    return concepts_df.values
