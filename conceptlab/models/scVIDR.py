import pertpy as pt
import scanpy as sc
import scvi
import hydra
import numpy as np

class scVIDR:
    def __init__(self, 
                 mod_cfg,
                 max_epochs = 1000,
                 batch_size = 128,
                 lr = 3e-4,
                 concept_key= "concepts",
                 concepts_to_flip: list = [],
                 concepts_as_cov = "",
                 concepts_to_flip_ref:list = [],
                 ct_key = "",
                 num_workers = 0,
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
        self.lr = lr
        self.concept_key = concept_key
        
        self.num_workers = num_workers
        self.concepts_to_flip = concepts_to_flip
        self.concepts_as_cov = concepts_as_cov
        self.concepts_to_flip_ref = concepts_to_flip_ref

        self.ct_key = ct_key

        self.mod_cfg = mod_cfg

    def train(self,adata_train):
        
        adata_train_ = adata_train.copy()
        adata_train_.obs["concepts_to_flip"] = adata_train_.obsm["concepts"][self.concepts_to_flip].astype(str).agg("_".join, axis = 1)
        #adata_train_.obs["concepts_as_cov"] = adata_train_.obsm["concepts"][self.concepts_as_cov].astype(str).agg("_".join, axis = 1)

        scvi.model.SCVI.setup_anndata(
            adata_train_,
            layer = "og",
            batch_key = "concepts_to_flip",
            labels_key= self.concepts_as_cov
        )

        self.model =self.mod_cfg(adata = adata_train_)

        self.model.train(max_epochs = self.max_epochs,
            early_stopping = True,
            early_stopping_patience = 25,
            plan_kwargs = {"lr": self.lr,
                           "weight_decay": 1e-6,
                           "optimizer": "Adam"})
        
        latent_train, sample_train = self.model.posterior_predictive_sample(adata_train_)

        latent_train = latent_train.cpu()
        # Collect the z_refs. 
        z_means_control = []
        z_means_treat = []

        unique_ct = adata_train.obs[self.concepts_as_cov].unique()
        for ct in unique_ct:
            latent_train_ct_mask = (adata_train.obs[self.concepts_as_cov] == ct)
            
            control_mask = 1
            for ic, ctf in enumerate(self.concepts_to_flip):
                control_mask *= adata_train_.obsm["concepts"][ctf]  == self.concepts_to_flip_ref[ic]
            
            treat_mask = 1
            for ic, ctf in enumerate(self.concepts_to_flip):
                treat_mask *= adata_train_.obsm["concepts"][ctf]  == 1-self.concepts_to_flip_ref[ic]

        
            latent_train_ct_control_mask  = latent_train_ct_mask * control_mask
            latent_train_ct_treat_mask = latent_train_ct_mask * treat_mask 

            z_ct_control = latent_train[latent_train_ct_control_mask]
            z_ct_treat = latent_train[latent_train_ct_treat_mask]

            if (len(z_ct_control)>0) and (len(z_ct_treat)>0):
                z_means_control.append(z_ct_control.mean(0))
                z_means_treat.append(z_ct_treat.mean(0))

        z_means_control = np.stack(z_means_control)
        z_means_treat = np.stack(z_means_treat)

        deltas = z_means_treat - z_means_control

        from sklearn.linear_model import LinearRegression

        self.reg = LinearRegression().fit(z_means_control, deltas)


    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip):
        
        adata_inter_ = adata_inter.copy()
        adata_inter_.obs["concepts_to_flip"] = adata_inter_.obsm["concepts"][self.concepts_to_flip].astype(str).agg("_".join, axis = 1)
        
        scvi.model.SCVI.setup_anndata(
            adata_inter_,
            layer = "og",
            batch_key = "concepts_to_flip",
            labels_key= self.concepts_as_cov
        )

        latent_inter, sample_inter = self.model.posterior_predictive_sample(adata_inter_)
        latent_inter = latent_inter.cpu()

        z_means_inter = latent_inter.mean(0)
        delta_pred = self.reg.predict(z_means_inter[None,:])

        latent_treated = latent_inter + delta_pred

        latent_treated, sample_treated = self.model.posterior_predictive_sample(adata_inter_, set_z = latent_treated)

        breakpoint()
        pred_adata = adata_inter_.copy()
        pred_adata.layers["og"] = sample_treated
        pred_adata.obs['ident'] = 'intervened on'

        return pred_adata