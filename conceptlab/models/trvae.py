import scanpy as sc
import hydra
import numpy as np
import trvaep

# based on: https://nbviewer.org/github/theislab/trvaep/blob/master/example/sample_notebook.ipynb


class trVAE:
    def __init__(
        self,
        lr=0.001,
        concept_key="concepts",
        concepts_to_flip: list = [],
        concepts_as_cov="",
        concepts_to_flip_ref: list = [],
        ct_key="",
        num_workers=0,
        target_sum=1000,
        train_frac=0.85,
        obsm_key="X",  # for consistency with other models (e.g. in evaluation)
        seed: int = 0,
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

        self.lr = lr
        self.concept_key = concept_key
        self.train_frac = train_frac
        self.seed = seed

        self.num_workers = num_workers
        self.concepts_to_flip = concepts_to_flip
        self.concepts_to_flip_ref = concepts_to_flip_ref

        self.ct_key = ct_key

        self.target_sum = target_sum
        self.obsm_key = obsm_key

    def train(self, adata_train):

        adata_train_ = adata_train.copy()
        adata_train_.obs["concepts_to_flip"] = (
            adata_train_.obsm["concepts"][self.concepts_to_flip]
            .astype(str)
            .agg("_".join, axis=1)
        )

        n_conditions = adata_train_.obs["concepts_to_flip"].nunique()

        self.model = trvaep.CVAE(
            adata_train_.n_vars,
            num_classes=n_conditions,
            encoder_layer_sizes=[128, 32],
            decoder_layer_sizes=[32, 128],
            latent_dim=10,
            alpha=0.0001,
            use_mmd=True,
            beta=10,
        )

        trainer = trvaep.Trainer(
            self.model,
            adata_train_,
            condition_key="concepts_to_flip",
            learning_rate=self.lr,
            train_frac=self.train_frac,
            seed=self.seed,
        )

        trainer.train_trvae(200, 512, early_patience=30)
        return

    def predict_intervention(
        self, adata_inter, hold_out_label, concepts_to_flip, values_to_set=None
    ):

        assert (
            concepts_to_flip == self.concepts_to_flip
        ), f"concepts to flip in prediction {concepts_to_flip} should be the same as in training {self.concepts_to_flip}"

        adata_inter_ = adata_inter.copy()
        adata_inter_.obs["concepts_to_flip"] = (
            adata_inter_.obsm["concepts"][self.concepts_to_flip]
            .astype(str)
            .agg("_".join, axis=1)
        )

        assert adata_inter_.obs["concepts_to_flip"].nunique() == 1

        predicted_data = self.model.predict(
            x=adata_inter_.X.A,
            y=adata_inter_.obs["concepts_to_flip"].tolist(),
            target=hold_out_label,
        )

        pred_adata = adata_inter_.copy()

        if self.obsm_key == "X":
            pred_adata = adata_inter_.X = predicted_data
        else:
            # TODO remove maybe
            pred_adata.obsm[self.obsm_key] = predicted_data

        pred_adata.obs["ident"] = "intervened on"

        return pred_adata
