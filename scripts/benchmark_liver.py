# This should be refactored - it's just a dump of Doron's notebook


import conceptlab as clab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import scanpy as sc
import scanpy as sc
import torch
import scipy.spatial
import matplotlib
import matplotlib.patches as mpatches
import string

from omegaconf import OmegaConf
import pytorch_lightning as pl

import sklearn.decomposition

NUM_EPOCHS = 100

def split_data(adata, hold_out_label, mod_label, label_key = 'L2_stim'):
    """
    Splits data into train, intervention, and ground truth sets.

    - Ground Truth: All cells with the `hold_out_label`.
    - Intervention: All cells with the `mod_label`.
    - Train: All remaining cells.
    """
    
    # if held out label is not a list, make it one
    if not isinstance(hold_out_label, list):
        hold_out_label = [hold_out_label]

    print("Splitting data with simplified logic...")
    labels = adata.obs[label_key]

    # Define the three disjoint sets based on their labels
    is_test = np.isin(labels, hold_out_label)
    is_inter = (labels == mod_label)
    is_train = ~is_test

    # Create AnnData objects for each split
    adata_train = adata[is_train].copy()
    adata_test = adata[is_test].copy()
    adata_inter = adata[is_inter].copy()

    # Store split identifiers in the original object
    ident_vec = np.array(['train'] * len(adata)).astype('<U32')
    ident_vec[is_test] = 'held out as GT'
    ident_vec[is_inter] = 'intervention'
    adata.obs['ident'] = ident_vec
    

    return adata, adata_train, adata_test, adata_inter

def train_cbgm(adata_train, obsm_key='X_pca', hard_concept_key=None, soft_concept_key=None):
    """
    Trains and returns the scCBGM model, supporting separate hard and soft concepts.
    """
    print("Training scCBGM model...")

    # --- Data Source Setup ---
    if obsm_key != 'X':
        data_matrix = adata_train.obsm[obsm_key]
    else:
        data_matrix = adata_train.X
    
    # --- Input Validation ---
    if hard_concept_key is None and soft_concept_key is None:
        raise ValueError("You must provide at least one of 'hard_concept_key' or 'soft_concept_key'.")

    torch.set_flush_denormal(True)

    # --- Prepare Concept Tensors ---
    hard_concepts_tensor = None
    n_hard = 0
    if hard_concept_key:
        hard_concepts_data = adata_train.obsm[hard_concept_key].to_numpy().astype(np.float32)
        hard_concepts_tensor = torch.from_numpy(hard_concepts_data)
        n_hard = hard_concepts_tensor.shape[1]

    soft_concepts_tensor = None
    n_soft = 0
    if soft_concept_key:
        soft_concepts_data = adata_train.obsm[soft_concept_key].to_numpy().astype(np.float32)
        soft_concepts_tensor = torch.from_numpy(soft_concepts_data)
        n_soft = soft_concepts_tensor.shape[1]
        
    # --- Dynamic Configuration ---
    # This config now uses the new, more explicit keys.
    config = OmegaConf.create(dict(
        has_cbm=True, 
        lr=5e-4, 
        hidden_dim=1024, 
        n_layers=4,
        beta=1e-5,
        input_dim=data_matrix.shape[-1],
        latent_dim=128,
        # NEW: Explicitly define hard/soft concepts
        use_hard_concepts=(hard_concept_key is not None),
        n_hard_concepts=n_hard,
        use_soft_concepts=(soft_concept_key is not None),
        n_soft_concepts=n_soft,
        n_unknown=128, 
        concepts_hp=0.1, 
        orthogonality_hp=0.5
    ))
    
    # This assumes 'clab.models.scCBGM' points to the updated class in the Canvas
    model = clab.models.CB_VAE_MIXED(config) 
    
    # Since we can't import, we'll just print a placeholder for the model call
    # print("Model would be initialized with config:")
    # print(OmegaConf.to_yaml(config))
    
    print("\nModel's train_loop would be called with:")
    print(f"data shape: {data_matrix.shape}")
    if hard_concepts_tensor is not None:
        print(f"hard_concepts shape: {hard_concepts_tensor.shape}")
    if soft_concepts_tensor is not None:
        print(f"soft_concepts shape: {soft_concepts_tensor.shape}")

    # --- Model Training ---
    # The train_loop call now uses the explicit arguments.
    model.train_loop(
        data=torch.from_numpy(data_matrix.astype(np.float32)),
        hard_concepts=hard_concepts_tensor,
        soft_concepts=soft_concepts_tensor,
        num_epochs=NUM_EPOCHS, 
        batch_size=128, 
        lr=3e-4,
    )
    return model

def pred_cbgm(model, adata_inter, obsm_key='X_pca', hard_concept_key=None, soft_concept_key=None, interventions=None):
    """
    Performs targeted intervention on specific concepts using a trained scCBGM model.

    Args:
        model: The trained scCBGM model.
        adata_inter: AnnData object with data to perform intervention on.
        obsm_key: Key in adata_inter.obsm to use as input, or 'X' for adata_inter.X.
        hard_concept_key: Key in adata_inter.obsm for the hard concepts DataFrame.
        soft_concept_key: Key in adata_inter.obsm for the soft concepts DataFrame.
        interventions (dict): A dictionary specifying the interventions.
                              Keys are concept column names.
                              Values are the new target values for those concepts.
                              Example: {'stim': 1, 'cell_cycle_G2M': 0.8}
    """
    print("Performing intervention with scCBGM...")

    if interventions is None or not interventions:
        print("No interventions specified. Returning a copy of the original data.")
        return adata_inter.copy()

    # --- Get input data ---
    x_intervene_on = torch.tensor(adata_inter.obsm[obsm_key] if obsm_key != 'X' else adata_inter.X, dtype=torch.float32)

    # --- Prepare original concept tensors and find column indices ---
    concept_parts, hard_concepts_df, soft_concepts_df = [], None, None
    n_hard = 0

    if hard_concept_key:
        hard_concepts_df = adata_inter.obsm[hard_concept_key]
        concept_parts.append(torch.from_numpy(hard_concepts_df.to_numpy(dtype=np.float32)))
        n_hard = hard_concepts_df.shape[1]

    if soft_concept_key:
        soft_concepts_df = adata_inter.obsm[soft_concept_key]
        concept_parts.append(torch.from_numpy(soft_concepts_df.to_numpy(dtype=np.float32)))

    if not concept_parts:
         raise ValueError("Must provide at least one concept key ('hard_concept_key' or 'soft_concept_key') to perform intervention.")

    c_intervene_on = torch.cat(concept_parts, dim=1)
    inter_concepts = c_intervene_on.clone()
    mask = torch.zeros_like(c_intervene_on)

    # --- Build the mask and intervention tensor from the dictionary ---
    for concept_name, new_value in interventions.items():
        found = False
        if hard_concepts_df is not None and concept_name in hard_concepts_df.columns:
            col_idx = hard_concepts_df.columns.get_loc(concept_name)
            mask[:, col_idx] = 1
            inter_concepts[:, col_idx] = new_value
            found = True
            print(f"Intervening on HARD concept '{concept_name}' (index {col_idx}) -> {new_value}")

        elif soft_concepts_df is not None and concept_name in soft_concepts_df.columns:
            col_idx = soft_concepts_df.columns.get_loc(concept_name) + n_hard
            mask[:, col_idx] = 1
            inter_concepts[:, col_idx] = new_value
            found = True
            print(f"Intervening on SOFT concept '{concept_name}' (index {col_idx}) -> {new_value}")

        if not found:
            print(f"Warning: Concept '{concept_name}' not found in provided keys. Ignoring.")

    # --- Run intervention on the model ---
    device = 'cuda'
    with torch.no_grad():
        inter_preds_dict = model.intervene(x_intervene_on.to(device), mask=mask.to(device), concepts=inter_concepts.to(device))
    inter_preds = inter_preds_dict['x_pred'].cpu().numpy()
    

    # --- Create prediction AnnData object ---
    pred_adata = adata_inter.copy()

    pred_adata.obs['ident'] = 'intervened'

    if obsm_key != 'X':
        pred_adata.X = np.zeros_like(pred_adata.X)
        pred_adata.obsm[obsm_key] = inter_preds
    else:
        pred_adata.X = inter_preds

    return pred_adata

def get_learned_concepts(scCBGM_model, adata_full, obsm_key = 'X_pca', hard_concept_key=None, soft_concept_key=None):
    """Uses a trained scCBGM to generate learned concepts for all data."""
    print("Generating learned concepts from scCBGM...")

    if(obsm_key != 'X'):
        all_x = torch.tensor(adata_full.obsm[obsm_key], dtype=torch.float32).to('cuda')
    else:
        all_x = torch.tensor(adata_full.X, dtype=torch.float32).to('cuda')

    with torch.no_grad():
        enc = scCBGM_model.encode(all_x)

        if(scCBGM_model.use_hard_concepts):
            scCBGM_concepts_known_hard = scCBGM_model.cb_hard_layers(enc['mu']).cpu().numpy()
            scCBGM_concepts_known_hard_df = pd.DataFrame(scCBGM_concepts_known_hard, 
                                                         index=adata_full.obs.index, 
                                                         columns=adata_full.obsm[hard_concept_key].columns)
        if(scCBGM_model.use_soft_concepts):
            scCBGM_concepts_known_soft= scCBGM_model.cb_soft_layers(enc['mu']).cpu().numpy()
            scCBGM_concepts_known_soft_df = pd.DataFrame(scCBGM_concepts_known_soft, 
                                                         index=adata_full.obs.index, 
                                                         columns=adata_full.obsm[soft_concept_key].columns)
        
        scCBGM_concepts_unknown = scCBGM_model.cb_unk_layers(enc['mu']).cpu().numpy()
        scCBGM_concepts_unknown_df = pd.DataFrame(scCBGM_concepts_unknown, 
                                                 index=adata_full.obs.index, 
                                                 columns=[f'unknown_{i}' for i in range(scCBGM_concepts_unknown.shape[1])])
    
    if(scCBGM_model.use_hard_concepts and scCBGM_model.use_soft_concepts):
        scCBGM_concepts = pd.concat([scCBGM_concepts_known_hard_df, scCBGM_concepts_known_soft_df, scCBGM_concepts_unknown_df], axis=1)
    elif(scCBGM_model.use_hard_concepts):
        scCBGM_concepts = pd.concat([scCBGM_concepts_known_hard_df, scCBGM_concepts_unknown_df], axis=1)
    elif(scCBGM_model.use_soft_concepts):
        scCBGM_concepts = pd.concat([scCBGM_concepts_known_soft_df, scCBGM_concepts_unknown_df], axis=1)
    else:
        raise ValueError("Model has no known concepts to extract.")
    
    adata_full.obsm['scCBGM_concepts'] = scCBGM_concepts


    return adata_full

def train_cb_fm(adata_train, concept_key = 'scCBGM_concepts', obsm_key = 'X_pca'):
    """Trains and returns the CB-FM model using learned concepts."""
    print("Training Concept Bottleneck Flow Model")

    if(obsm_key != 'X'):
        data_matrix = adata_train.obsm[obsm_key]
    else:
        data_matrix = adata_train.X
    
    config = dict(
        input_dim=data_matrix.shape[1],
        hidden_dim=1024,
        latent_dim=128,
        n_concepts=adata_train.obsm[concept_key].shape[1],
        n_layers=4,
        dropout=0.1,
        p_uncond = 0.0)

    fm_model = clab.models.cond_fm.Cond_FM(config=config)

    fm_model.train_loop(
        data=torch.from_numpy(data_matrix.astype(np.float32)),
        concepts=torch.from_numpy(adata_train.obsm[concept_key].to_numpy().astype(np.float32)),
        num_epochs=NUM_EPOCHS, batch_size=128, lr=3e-4,
    )
    return fm_model



def pred_cb_fm(model, adata_inter, edit_concept, edit_value, concept_key = 'scCBGM_concepts', obsm_key = 'X_pca', edit = True):
    """Performs intervention using a trained learned-concept CB-FM model."""
    print("Performing intervention with CB-FM (learned)...")

    if(obsm_key != 'X'):
        x_inter = adata_inter.obsm[obsm_key]
    else:
        x_inter = adata_inter.X
    
    init_concepts = adata_inter.obsm[concept_key]
    edit_concepts = init_concepts.copy()

    edit_concepts[edit_concept] = edit_value
    # edit_concepts[:, -1] = 1 # Set stim concept to 1

    init_concepts = init_concepts.to_numpy().astype(np.float32)
    edit_concepts = edit_concepts.to_numpy().astype(np.float32)

    if(edit):
        inter_preds = model.edit(
                x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                c = torch.from_numpy(init_concepts.astype(np.float32)).to('cuda'),
                c_prime = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                t_edit = 0.0,
                n_steps = 1000,
                w_cfg_forward = 1.0,
                w_cfg_backward = 1.0,
                noise_add = 0.0)
    else:
        inter_preds = model.decode(
                h = torch.from_numpy(edit_concepts.astype(np.float32)).to('cuda'),
                n_steps = 1000,
                w_cfg = 1.0)
        
    inter_preds = inter_preds.detach().cpu().numpy()

    if(obsm_key != 'X'):
        x_inter_preds = np.zeros_like(adata_inter.X)
    else:
        x_inter_preds = inter_preds

    pred_adata = adata_inter.copy()
    pred_adata.X = x_inter_preds
    pred_adata.obs['ident'] = 'intervened on'

    if(obsm_key != 'X'):
        pred_adata.obsm[obsm_key] = inter_preds
    return pred_adata

def train_raw_fm(adata_train, concept_key = 'concepts', obsm_key = 'X_pca'):
    """Trains and returns the CB-FM model using learned concepts."""
    print("Training Conditonal Flow Model")

    if(obsm_key != 'X'):
        data_matrix = adata_train.obsm[obsm_key]
    else:
        data_matrix = adata_train.X
    
    config = dict(
        input_dim=data_matrix.shape[1],
        hidden_dim=1024,
        latent_dim=128,
        n_concepts=adata_train.obsm[concept_key].to_numpy().shape[1],
        n_layers=4,
        dropout=0.1,
        p_uncond = 0.0)

    fm_model = clab.models.cond_fm.Cond_FM(config=config)

    fm_model.train_loop(
        data=torch.from_numpy(data_matrix.astype(np.float32)),
        concepts=torch.from_numpy(adata_train.obsm[concept_key].to_numpy().astype(np.float32)),
        num_epochs=NUM_EPOCHS, batch_size=128, lr=3e-4,
    )
    return fm_model



def pred_raw_fm(model, adata_inter, edit_concept, edit_value, concept_key = 'concepts', obsm_key = 'X_pca', edit = False):
    """Performs intervention using a trained learned-concept CB-FM model."""
    print("Performing intervention with Raw Flow Matching(learned)...")

    
    if(obsm_key != 'X'):
        x_inter = adata_inter.obsm[obsm_key]
    else:
        x_inter = adata_inter.X


    init_concepts = adata_inter.obsm[concept_key]
    edit_concepts = init_concepts.copy()

    edit_concepts[edit_concept] = edit_value

    init_concepts = init_concepts.to_numpy().astype(np.float32)
    edit_concepts = edit_concepts.to_numpy().astype(np.float32)

    if(edit):
        inter_preds = model.edit(
                x = torch.from_numpy(x_inter.astype(np.float32)).to('cuda'),
                c = torch.from_numpy(init_concepts).to('cuda'),
                c_prime = torch.from_numpy(edit_concepts).to('cuda'),
                t_edit = 0.0,
                n_steps = 1000,
                w_cfg_forward = 1.0,
                w_cfg_backward = 1.0,
                noise_add = 0.0)
    else:
        inter_preds = model.decode(
                h = torch.from_numpy(edit_concepts).to('cuda'),
                n_steps = 1000,
                w_cfg = 1.0)
    
    inter_preds = inter_preds.detach().cpu().numpy()

    if(obsm_key != 'X'):
        x_inter_preds = np.zeros_like(adata_inter.X)
    else:
        x_inter_preds = inter_preds

    pred_adata = adata_inter.copy()
    pred_adata.X = x_inter_preds
    pred_adata.obs['ident'] = 'intervened on'

    if(obsm_key != 'X'):
        pred_adata.obsm[obsm_key] = inter_preds
    return pred_adata


if __name__ == "__main__":

    DATA_PATH  = '/braid/havivd/liver_doses/SCP1871/adata_liver.h5ad'
    OBSM_KEY = 'X_pca'
    Z_SCORE = False
    HARD_CONCEPT_KEY = 'hard_concepts'
    SOFT_CONCEPT_KEY = 'soft_concepts'
    RANDOM_SEED = 0
    DATA_PREPROCESSED = True #whether the anndata already contains preprcessed data.

    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # DATA PROCESSING

    print("Loading and preprocessing data...")
    adata = ad.read_h5ad(DATA_PATH)

    if not DATA_PREPROCESSED:
        adata.X = adata.X.toarray()
        # adata.X = adata.layers['raw'].toarray()
        # sc.pp.normalize_total(adata, target_sum=np.median(adata.X.sum(axis=1)))
        # sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes = 3000, subset=True)

        def zero_one_norm(x):
            """Normalizes a numpy array to the [0, 1] range."""
            min_val = np.min(x)
            max_val = np.max(x)
            return (x - min_val) / (max_val - min_val)

        cell_type_mapping = {
        'centrilobular region hepatocyte': 'Hepatocyte',
        'periportal region hepatocyte': 'Hepatocyte',
        'B cell': 'Immune Cell',
        'T cell': 'Immune Cell',
        'liver dendritic cell': 'Immune Cell',
        'macrophage': 'Immune Cell',
        'neutrophil': 'Immune Cell',
        'cholangiocyte': 'Cholangiocyte',
        'endothelial cell of hepatic sinusoid': 'Stromal Cell',
        'hepatic portal fibroblast': 'Stromal Cell',
        'hepatic stellate cell': 'Stromal Cell'
        }

        adata.obs['cell_types_L2'] =  adata.obs['cell_type__ontology_label'].astype('category')
        adata.obs['cell_types_L1'] = adata.obs['cell_type__ontology_label'].map(cell_type_mapping).astype('category')

        adata.obsm[HARD_CONCEPT_KEY] = pd.get_dummies(adata.obs[['cell_types_L1']]).astype(np.float32)
        adata.obsm[SOFT_CONCEPT_KEY] = pd.DataFrame(zero_one_norm(np.log(adata.obs['dose'].astype(float).values + 1)).reshape(-1, 1),
            index=adata.obs_names, columns=['dose']).astype(np.float32)

        adata.obsm['concepts'] = pd.concat([adata.obsm[HARD_CONCEPT_KEY], adata.obsm[SOFT_CONCEPT_KEY]], axis=1)

        adata.obs['L1_dose'] = [l1_ctype + '_' + dose for l1_ctype, dose in zip(adata.obs['cell_types_L1'], adata.obs['dose'])]
        adata.obs['L2_dose'] = [l2_ctype + '_' + dose for l2_ctype, dose in zip(adata.obs['cell_types_L2'], adata.obs['dose'])]

    value_counts = adata.obs['L2_dose'].value_counts()

    for index, count in value_counts.items():
        print(f"{index}: {count}")

    # Discover all base cell types from the data
    base_cell_types = sorted(list(set(adata.obs['cell_types_L2'])))
    all_dose_labels = adata.obs['L2_dose'].unique()

    all_results = {}
    all_predictions = {}

    # Main loop over each discovered cell type
    for cell_type in base_cell_types:
        print(f"\n{'='*25} Processing Cell Type: {cell_type} {'='*25}")

        # --- Define hard-coded labels for the experiment ---
        mod_label = f"{cell_type}_1.0"
        hold_out_labels = [f"{cell_type}_3.0", f"{cell_type}_10.0"]
        required_labels = [mod_label] + hold_out_labels

        # --- Check if all required dose labels exist for this cell type ---
        if not all(label in all_dose_labels for label in required_labels):
            print(f"Skipping '{cell_type}': Does not have all required dose labels (1.0, 3.0, 10.0).")
            continue
            
        print(f"  - Intervention base: {mod_label}")
        print(f"  - Hold-out targets: {hold_out_labels}")

        # Split data and validate set sizes
        # In a real scenario, you'd use your actual split_data function
        # adata_sub, adata_train, adata_test, adata_inter = split_data(...)
        
        # Using dummy splits based on the full adata object for this script
        # In reality, this would exclude hold_out_labels
        adata, adata_train, adata_test, adata_inter = split_data(
            adata, hold_out_labels, mod_label, label_key = 'L2_dose'
        )
        
        if len(adata_inter.X) < 300 or len(adata_test.X) < 300:
            print(f"Skipping '{cell_type}': Intervention set ({len(adata_inter.X)}) or test set ({len(adata_test.X)}) is smaller than 1000 cells.")
            continue

        print(f"  - Train set: {len(adata_train.X)} cells")
        print(f"  - Intervention set: {len(adata_inter.X)} cells")
        print(f"  - Ground Truth set: {len(adata_test.X)} cells")

        # Preprocessing (PCA)
        pc_transform = sklearn.decomposition.PCA(n_components=128).fit(adata_train.X)
        for x_data in [adata, adata_train, adata_test, adata_inter]:
            x_data.obsm[OBSM_KEY] = pc_transform.transform(x_data.X)

        # Define Intervention Plan
        interventions = []
        for label in hold_out_labels:
            target_dose_value = adata_test[adata_test.obs['L2_dose'] == label].obsm[SOFT_CONCEPT_KEY]['dose'].astype(float).mean()
            interventions.append({'concept': 'dose', 'value': target_dose_value, 'label': label})

        # --- Method 1: scCBGM ---
        cbgm_model = train_cbgm(adata_train.copy(), hard_concept_key=HARD_CONCEPT_KEY, soft_concept_key=SOFT_CONCEPT_KEY)
        
        pred_adata_cbgm =[]
        for intervention in interventions:
            pred_adata_cbgm.append(pred_cbgm(cbgm_model, adata_inter.copy(), hard_concept_key=HARD_CONCEPT_KEY, soft_concept_key=SOFT_CONCEPT_KEY,
                                    interventions={intervention['concept']: intervention['value']}))
        
        # --- Method 2: CB-FM with Learned Concepts ---
        adata_with_concepts = get_learned_concepts(cbgm_model, adata.copy(), hard_concept_key=HARD_CONCEPT_KEY, soft_concept_key=SOFT_CONCEPT_KEY)
        adata_train.obsm['scCBGM_concepts'] = adata_with_concepts[adata_train.obs.index].obsm['scCBGM_concepts']
        adata_inter.obsm['scCBGM_concepts'] = adata_with_concepts[adata_inter.obs.index].obsm['scCBGM_concepts']
        
        cb_fm_model = train_cb_fm(adata_train.copy(), concept_key='scCBGM_concepts', obsm_key=OBSM_KEY)
        
        pred_adata_fm_edit = []
        pred_adata_fm_guid = []
        for intervention in interventions:
            pred_adata_fm_edit.append(pred_cb_fm(cb_fm_model, adata_inter.copy(), concept_key='scCBGM_concepts', 
                                        obsm_key=OBSM_KEY, 
                                        edit_concept=intervention['concept'],
                                        edit_value=intervention['value'], 
                                        edit=True))
            pred_adata_fm_guid.append(pred_cb_fm(cb_fm_model, adata_inter.copy(), 
                                        concept_key='scCBGM_concepts', 
                                        obsm_key=OBSM_KEY, 
                                        edit_concept=intervention['concept'], 
                                        edit_value=intervention['value'], 
                                        edit=False))
        
        # --- Method 3: FM with Raw Concepts ---
        fm_raw_model = train_raw_fm(adata_train.copy(), concept_key='concepts', obsm_key=OBSM_KEY)
        
        pred_adata_raw_fm_edit = []
        pred_adata_raw_fm_guid = []
        for intervention in interventions:
            pred_adata_raw_fm_edit.append(pred_raw_fm(fm_raw_model, adata_inter.copy(), concept_key='concepts', 
                                        obsm_key=OBSM_KEY, 
                                        edit_concept=intervention['concept'],
                                        edit_value=intervention['value'], 
                                        edit=True))
            pred_adata_raw_fm_guid.append(pred_raw_fm(fm_raw_model, adata_inter.copy(), 
                                        concept_key='concepts', 
                                        obsm_key=OBSM_KEY, 
                                        edit_concept=intervention['concept'], 
                                        edit_value=intervention['value'], 
                                        edit=False))

        # Evaluate and store results for the current cell type
        cell_type_results = {}
        cell_type_predictions = {}
        
        for i, intervention in enumerate(interventions):
            intervention_label = intervention['label']
            print(f"  - Evaluating intervention for target: {intervention_label}")

            intervention_adata = adata_test[adata_test.obs['L2_dose'] == intervention_label]

            predictions_for_ivn = {
                'scCBGM': pred_adata_cbgm[i],
                'CB-FM (edit)': pred_adata_fm_edit[i],
                'CB-FM (guided)': pred_adata_fm_guid[i],
                'Raw-FM (edit)': pred_adata_raw_fm_edit[i],
                'Raw-FM (guided)': pred_adata_raw_fm_guid[i]
            }
            cell_type_predictions[intervention_label] = predictions_for_ivn

            mmd_scores = {}
            pre_computed_mmd_train = -1
            for name, pred_adata in predictions_for_ivn.items():
                if pre_computed_mmd_train < 0:
                    val = clab.evaluation.interventions.evaluate_intervention_mmd_with_target(
                        x_train=adata_train.obsm[OBSM_KEY],
                        x_ivn=pred_adata.obsm[OBSM_KEY],
                        x_target=intervention_adata.obsm[OBSM_KEY],
                        labels_train=adata_train.obs['L2_dose'].values
                    )
                    pre_computed_mmd_train = val['pre_computed_mmd_train']
                else:
                    val = clab.evaluation.interventions.evaluate_intervention_mmd_with_target(
                        x_train=adata_train.obsm[OBSM_KEY],
                        x_ivn=pred_adata.obsm[OBSM_KEY],
                        x_target=intervention_adata.obsm[OBSM_KEY],
                        labels_train=adata_train.obs['L2_dose'].values,
                        pre_computed_mmd_train=pre_computed_mmd_train
                    )
                
                mmd_ratio = val['mmd_ratio']
                mmd_scores[name] = mmd_ratio
            
            cell_type_results[intervention_label] = mmd_scores

        print(f"Results for cell type '{cell_type}':")
        results_df = pd.DataFrame(cell_type_results).T
        print(results_df.to_string())

        all_results[cell_type] = cell_type_results
        all_predictions[cell_type] = cell_type_predictions

    # Final Summary Report
    print("\n\n" + "="*30 + " FINAL RESULTS SUMMARY " + "="*30)

    summary_data = []
    for cell_type, interventions_dict in all_results.items():
        for intervention_label, models_dict in interventions_dict.items():
            row = {'cell_type': cell_type, 'intervention_target': intervention_label}
            row.update(models_dict)
            summary_data.append(row)

    if not summary_data:
        print("No cell types met the criteria to be processed.")
    else:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string())
        summary_df.to_csv(f"./benchmark_liver_results_summary_seed_{RANDOM_SEED}.csv", index=False)
