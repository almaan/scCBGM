
import numpy as np


def split_data_for_counterfactuals(adata, hold_out_label, mod_label, label_variable, p_intervention=0.2):
    """Splits data into train, intervention, and ground truth sets."""
    print("Splitting data for counterfactual experiment...")
    
    labels = adata.obs[label_variable]
    is_test = (labels == hold_out_label)
    is_inter_pool = (labels == mod_label)

    # Create a random mask to select a subset for intervention
    inter_mask = np.random.binomial(1, p=p_intervention, size=is_inter_pool.sum()).astype(bool)
    is_inter = np.zeros_like(labels, dtype=bool)
    is_inter[is_inter_pool] = inter_mask
    
    is_train = ~is_test & ~is_inter

    # Create AnnData objects for each split
    adata_train = adata[is_train].copy()
    adata_test = adata[is_test].copy()
    adata_inter = adata[is_inter].copy()

    # Store split identifiers in the original object for later merging
    ident_vec = np.array(['train'] * len(adata)).astype('<U32')
    ident_vec[is_test] = 'held out as GT'
    ident_vec[is_inter] = 'held out for intervention'
    adata.obs['ident'] = ident_vec
    
    print(f"Train set: {len(adata_train)} cells")
    print(f"Intervention set: {len(adata_inter)} cells")
    print(f"Ground Truth set: {len(adata_test)} cells")

    return adata, adata_train, adata_test, adata_inter
