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
import argparse

from omegaconf import OmegaConf
import pytorch_lightning as pl

import sklearn.decomposition

from conceptlab.models.cinemaot import CinemaOT
from conceptlab.models.scgen import scGEN
from conceptlab.models.cem_vae import CEM_MetaTrainer
from conceptlab.models.biolord import Biolord

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


def main(methods_to_eval: list = ['cinemaot', 'scgen']):

    DATA_PATH  = '/braid/havivd/liver_doses/SCP1871/adata_liver.h5ad'
    RAW_DATA_PATH = '/braid/havivd/liver_doses/SCP1871/adata_liver_raw.h5ad'
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
    if "scgen" in methods_to_eval:
        # invert log1p
        adata.layers["og"] = np.floor(np.expm1(adata.X))
    
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
            x_data.uns['pc_transform'] = pc_transform

        # Define Intervention Plan
        interventions = []
        for label in hold_out_labels:
            target_dose_value = adata_test[adata_test.obs['L2_dose'] == label].obsm[SOFT_CONCEPT_KEY]['dose'].astype(float).mean()
            interventions.append({'concept': 'dose', 'value': target_dose_value, 'label': label})

        if "cinemaot" in methods_to_eval:
            # ---- CinemaOT Baseline ----
            cinema_ot_model = CinemaOT(thresh = 0.5,
                                        eps = 0.001,
                                        concept_key = "concepts",
                                        obsm_key = OBSM_KEY) #not used in CinemaOT - would always work
            cinema_ot_model.train(adata_train.copy())
            pred_adata_cot = []
            for intervention in interventions:
                pred_adata_cot.append(cinema_ot_model.predict_intervention(adata_inter.copy(),
                                                    hold_out_label = "dose",
                                                    concepts_to_flip = [intervention["concept"]],
                                                    values_to_set = [intervention["value"]]))
            
        # ---- End CinemaOT Baseline ----

        if "scgen" in methods_to_eval:
            # ----- ScGen Baseline -----
            scgen_model = scGEN(
                max_epochs=100,
                lr=1e-5,
                concept_key="concepts",
                concepts_to_flip=["dose"],
                concepts_as_cov="cell_types_L1",
                obsm_key=OBSM_KEY,
                num_workers = 6,
            )
            scgen_model.train(adata_train.copy())
            pred_adata_scgen = []
            for intervention in interventions:
                pred_adata_scgen.append(scgen_model.predict_intervention(adata_inter.copy(),
                                                    hold_out_label = "dose",
                                                    concepts_to_flip = [intervention["concept"]],
                                                    values_to_set = [intervention["value"]]))
            # ----- End ScGen Baseline -----

        if "cemvae" in methods_to_eval:
            cemvae_model = CEM_MetaTrainer(
                max_epochs= 200,
                concept_key = "concepts",
                num_workers = 4,
                obsm_key = OBSM_KEY,
                batch_size = 128,
                cbm_config ={"has_cbm": True,
                             "lr": 3e-4,
                             "n_layers": 4,
                             "hidden_size": 1024,
                             "beta": 1e-5,
                             "latent_dim": 128,
                             "n_unknown": 128,
                             "min_bottleneck_size": 128,
                             "concepts_hp": 0.1,
                             "orthogonality_hp": 0.5,
                             "use_soft_concepts": False}
                )
            cemvae_model.train(adata_train.copy())

        if "biolord" in methods_to_eval:
            
            assert len(adata_inter.obs["dose"].unique()) == 1
            concepts_to_flip_ref = [adata_inter.obs["dose"].unique()[0]]
            biolord_model = Biolord(
                concept_key= "concepts",
                concepts_to_flip= ["dose"],
                concepts_as_cov = "cell_types_L1",
                concepts_to_flip_ref= concepts_to_flip_ref,
                max_epochs=100,
                num_workers=4,
                n_latent=32,
                target_sum=1000,
                obsm_key=OBSM_KEY, # not used in BioLoRD - would always work on (normalized - log1p) count space.
                mod_cfg={
                    "decoder_width": 1024,
                    "decoder_depth": 4,
                    "attribute_nn_width": 512,
                    "attribute_nn_depth": 2,
                    "n_latent_attribute_categorical": 4,
                    "gene_likelihood": "normal",
                    "reconstruction_penalty": 1e2,
                    "unknown_attribute_penalty": 1e1,
                    "unknown_attribute_noise_param": 1e-1,
                    "attribute_dropout_rate": 0.1,
                    "use_batch_norm": False,
                    "use_layer_norm": False,
                    "seed": 42
                },
                trainer_cfg={
                    "n_epochs_warmup": 0,
                    "latent_lr": 1e-4,
                    "latent_wd": 1e-4,
                    "decoder_lr": 1e-4,
                    "decoder_wd": 1e-4,
                    "attribute_nn_lr": 1e-2,
                    "attribute_nn_wd": 4e-8,
                    "step_size_lr": 45,
                    "cosine_scheduler": True,
                    "scheduler_final_lr": 1e-5
                }
            )
            try:
                biolord_model.train(adata_train.copy())
                pred_adata_biolord = []
                for intervention in interventions:
                    pred_adata_biolord.append(biolord_model.predict_intervention(adata_inter.copy(),
                                                        hold_out_label = "dose",
                                                        concepts_to_flip = [intervention["concept"]],
                                                        values_to_set = [intervention["value"]]))
            except:
                pred_adata_biolord = [None] * len(interventions)
                print(f"Biolord failed for cell type {cell_type}, skipping.")

        # Evaluate and store results for the current cell type
        cell_type_results = {}
        cell_type_predictions = {}
        
        for i, intervention in enumerate(interventions):
            intervention_label = intervention['label']
            print(f"  - Evaluating intervention for target: {intervention_label}")

            intervention_adata = adata_test[adata_test.obs['L2_dose'] == intervention_label]

            predictions_for_ivn = {}
            if "scgen" in methods_to_eval:
                predictions_for_ivn['scgen'] = pred_adata_scgen[i]
            if "cinemaot" in methods_to_eval:
                predictions_for_ivn["cinemaot"] = pred_adata_cot[i]
            if "biolord" in methods_to_eval and pred_adata_biolord[i] is not None:
                predictions_for_ivn["biolord"] = pred_adata_biolord[i]
            
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
        methods_str = "_".join(methods_to_eval)
        summary_df.to_csv(f"./benchmark_liver_results_summary_baselines_{methods_str}_seed_{RANDOM_SEED}.csv", index=False)

if __name__ == "__main__":
    # use argparse for the methods to eval. Options are 'cinemaot' and 'scgen'
    parser = argparse.ArgumentParser(description="Benchmark liver baselines.")
    parser.add_argument("--cinemaot", action="store_true", help="Evaluate CinemaOT baseline")
    parser.add_argument("--scgen", action="store_true", help="Evaluate scGEN baseline")
    parser.add_argument("--cemvae", action="store_true", help="Evaluate CEVAE baseline (not implemented)")
    parser.add_argument("--biolord", action="store_true", help="Evaluate Biolord baseline (not implemented)")
    args = parser.parse_args()

    methods = []    
    if args.cinemaot:
        methods.append("cinemaot")
    if args.scgen:
        methods.append("scgen")
    if args.cemvae:
        methods.append("cemvae")
    if args.biolord:
        methods.append("biolord")

    main(methods_to_eval=methods)