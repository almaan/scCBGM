import pertpy as pt
import scanpy as sc
import biolord
import hydra
import numpy as np
import pandas as pd
import conceptlab as clab
import torch


class Biolord:
    def __init__(
        self,
        mod_cfg,
        trainer_cfg,
        max_epochs=1000,
        batch_size=128,
        n_latent=32,
        concept_key="concepts",
        concepts_to_flip: list = [],
        concepts_as_cov="",
        concepts_to_flip_ref: list = [],
        ct_key="",
        num_workers=0,
        target_sum=1000,
        obsm_key: str = "X",  # for consistency with other models (e.g. in evaluation)
        fm_mod_cfg=None,  # Config for FM decoder (no FM decoder if None)
        z_score: bool = False,  # whether to z-score data for FM
    ):
        """
        Class to train and predict interventions with a scVIDR model
        Inputs:
        - cbm_mod: the scVIDR model underlying the model
        - num_epochs: max number of epochs to train for
        - concept_key: Key in adata.obsm where the concept vectors are stored
        - num_workers: num workers in the dataloaders
        - pca: whether to train, and predict in PCA space.
        - zscore: whether to whiten the data
        - raw: whether to use "CellFlow" style - using only raw concepts
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.concept_key = concept_key

        self.num_workers = num_workers
        self.concepts_to_flip = concepts_to_flip
        self.concepts_as_cov = concepts_as_cov
        self.concepts_to_flip_ref = concepts_to_flip_ref

        self.ct_key = ct_key

        self.target_sum = target_sum

        self.mod_cfg = hydra.utils.instantiate(mod_cfg)
        self.trainer_cfg = hydra.utils.instantiate(trainer_cfg)

        self.fm_mod_cfg = fm_mod_cfg

        self.model = None
        self.fm_model = None

        self.obsm_key = obsm_key  # Only used when doing FM

    def train(self, adata_train):

        adata_train_ = adata_train.copy()

        ## Concat concepts in the obs and change name (append "concept_" to avoid confusion with other keys)
        observed_concept_names_ = adata_train_.obsm[self.concept_key].columns.tolist()
        observed_concept_names = ["concept_" + c for c in observed_concept_names_]
        concept_names_map = dict(zip(observed_concept_names_, observed_concept_names))
        concepts_df = (
            adata_train_.obsm[self.concept_key].copy().rename(columns=concept_names_map)
        )

        adata_train_.obs = pd.concat((adata_train_.obs, concepts_df), axis=1)

        biolord.Biolord.setup_anndata(
            adata_train_,
            ordered_attributes_keys=None,
            categorical_attributes_keys=observed_concept_names,
        )

        self.model = biolord.Biolord(
            adata=adata_train_,
            n_latent=self.n_latent,
            model_name="spatio_temporal_infected",
            module_params=self.mod_cfg,
            train_classifiers=False,
            # split_key="split_random",
        )

        self.model.train(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            plan_kwargs=self.trainer_cfg,
            early_stopping=True,
            early_stopping_patience=20,
            check_val_every_n_epoch=10,
            num_workers=self.num_workers,
            enable_checkpointing=False,
        )

        self.adata_train = adata_train_

        if self.fm_mod_cfg is not None:
            self.train_flow_matching()

        return

    def train_flow_matching(self):
        print("Training flow matching decoder...")
        if not self.model:
            raise ValueError(
                "Biolord Model has not been trained yet. Please train the model before training flow matching decoder."
            )

        fm_mod_cfg = hydra.utils.instantiate(self.fm_mod_cfg)
        _, concept_hiddens = self.model.get_latent_representation_adata(
            self.adata_train
        )
        concept_hiddens = concept_hiddens.X

        if self.obsm_key != "X":
            data_matrix = self.adata_train.obsm[self.obsm_key]
        else:
            data_matrix = self.adata_train.X
            if self.z_score:
                data_matrix = (
                    data_matrix - self.adata_train.var["mean"].to_numpy()[None, :]
                ) / self.adata_train.var["std"].to_numpy()[None, :]

        fm_mod_cfg["input_dim"] = data_matrix.shape[1]
        fm_mod_cfg["n_concepts"] = concept_hiddens.shape[1]
        self.fm_model = clab.models.cond_fm.Cond_FM(config=fm_mod_cfg)

        self.fm_model.train_loop(
            data=torch.from_numpy(data_matrix.astype(np.float32)),
            concepts=torch.from_numpy(concept_hiddens.astype(np.float32)),
            num_epochs=fm_mod_cfg["num_epochs"],
            batch_size=fm_mod_cfg["batch_size"],
            lr=fm_mod_cfg["lr"],
            num_workers=self.num_workers,
        )

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):

        assert (
            concepts_to_flip == self.concepts_to_flip
        ), f"concepts to flip in prediction {concepts_to_flip} should be the same as in training {self.concepts_to_flip}"

        adata_inter_ = adata_inter.copy()

        ## Concat concepts in the obs and change name (append "concept_" to avoid confusion with other keys)
        observed_concept_names_ = adata_inter_.obsm[self.concept_key].columns.tolist()
        observed_concept_names = ["concept_" + c for c in observed_concept_names_]
        concept_names_map = dict(zip(observed_concept_names_, observed_concept_names))
        concepts_df = (
            adata_inter_.obsm[self.concept_key].copy().rename(columns=concept_names_map)
        )

        adata_inter_.obs = pd.concat((adata_inter_.obs, concepts_df), axis=1)

        biolord.Biolord.setup_anndata(
            adata_inter_,
            ordered_attributes_keys=None,
            categorical_attributes_keys=observed_concept_names,
        )

        renamed_concepts_to_flip = ["concept_" + c for c in concepts_to_flip]
        concepts_to_flip_map = dict(zip(concepts_to_flip, renamed_concepts_to_flip))

        if self.fm_model is not None:
            _, latent_preds = self.model.get_latent_representation_adata(adata_inter_)
            latent_preds = latent_preds.X
            pred_adata = adata_inter_.copy()

            inter_preds = (
                self.fm_model.decode(
                    h=torch.from_numpy(latent_preds.astype(np.float32)).to("cuda"),
                    n_steps=1000,
                    w_cfg=1.0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            if self.obsm_key != "X":
                x_inter_preds = np.zeros_like(adata_inter.X)
            else:
                x_inter_preds = inter_preds
                if self.z_score:
                    x_inter_preds = (
                        x_inter_preds * adata_inter.var["std"].to_numpy()[None, :]
                    ) + adata_inter.var["mean"].to_numpy()[None, :]

            pred_adata = adata_inter.copy()
            pred_adata.X = x_inter_preds
            pred_adata.obs["ident"] = "intervened on"

            if self.obsm_key != "X":
                pred_adata.obsm[self.obsm_key] = inter_preds
            return pred_adata

        else:  # Classic Biolord (always in X space)
            adata_preds = self.model.compute_prediction_adata(
                self.adata_train,
                adata_inter_,
                target_attributes=renamed_concepts_to_flip,
            )

            pred_mask = 1
            for i_c, c in enumerate(concepts_to_flip):
                if values_to_set is not None:
                    pred_mask *= (
                        adata_preds.obs[concepts_to_flip_map[c]].values
                        == values_to_set[i_c]
                    )
                else:
                    pred_mask *= (
                        adata_preds.obs[concepts_to_flip_map[c]].values
                        == 1 - self.concepts_to_flip_ref[i_c]
                    )

            adata_preds_ = adata_preds[pred_mask == 1].copy()

            pred_adata = adata_inter_.copy()
            pred_adata.X = np.clip(adata_preds_.X, a_min=0, a_max=np.inf)
            pred_adata.obsm["X_pca"] = pred_adata.uns["pc_transform"].transform(
                pred_adata.X
            )

            pred_adata.obs["ident"] = "intervened on"

            return pred_adata
