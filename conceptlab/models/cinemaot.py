import pertpy as pt
import scanpy as sc


class CinemaOT:
    def __init__(
        self,
        thresh: float,
        eps: float,
        concept_key: str,
        obsm_key: str = "X",  # for consistency with other models (e.g. in evaluation)
    ):
        self.thresh = thresh
        self.eps = eps

        self.concept_key = concept_key
        self.obsm_key = obsm_key

    def train(self, adata_train):
        self.adata_train = adata_train

    def predict_intervention(self, adata_inter, hold_out_label, concepts_to_flip):

        # Infering the control signature - how to define a control
        control_signature = {
            concept: adata_inter.obsm["concepts"][concept].unique()
            for concept in concepts_to_flip
        }
        for k, v in control_signature.items():
            assert len(v) == 1
        control_signature = {k: v[0] for k, v in control_signature.items()}

        adata_cat = self.adata_train.concatenate(adata_inter)
        # adata_cat.obs["stim"] = adata_cat.obsm["concepts"]["stim"]
        adata_cat.obs["split_col"] = ["train"] * len(self.adata_train) + ["int"] * len(
            adata_inter
        )

        treated_mask = 1
        non_treated_mask = 1
        for k, v in control_signature.items():
            treated_mask *= adata_cat.obsm[self.concept_key][k].values == 1 - v
            non_treated_mask *= adata_cat.obsm[self.concept_key][k].values == v

        adata_cat.obs["treated"] = treated_mask
        adata_cat.obs["non_treated"] = non_treated_mask

        # Dataset of treated and non-treated only
        adata_t_not = adata_cat[
            (adata_cat.obs["treated"] == 1) | (adata_cat.obs["non_treated"] == 1)
        ].copy()

        assert (
            adata_t_not.obs["treated"].values
            == 1 - adata_t_not.obs["non_treated"].values
        ).all()

        cot = pt.tl.Cinemaot()
        sc.pp.pca(adata_t_not)

        de = cot.causaleffect(
            adata_t_not,
            pert_key="treated",
            control=0,
            return_matching=True,
            thres=self.thresh,
            smoothness=1e-5,
            eps=self.eps,
            solver="Sinkhorn",
        )

        ot_map = de.obsm["ot"]  # treatment vs controls

        adata_control = adata_t_not[adata_t_not.obs["non_treated"] == 1]
        adata_treated = adata_t_not[adata_t_not.obs["treated"] == 1]

        test_controls_idx = (adata_control.obs["split_col"].values == "int") & (
            adata_control.obs["non_treated"] == 1
        )

        ot_map_test = ot_map[:, test_controls_idx].T
        ot_map_test /= ot_map_test.sum(1).mean()

        adata_test_control = adata_control[
            test_controls_idx
        ]  # control units to predict

        adata_test_predicted = adata_control[test_controls_idx].copy()
        adata_test_predicted.X = (
            ot_map_test @ adata_treated.X
        )  # predicted treated units

        adata_test_predicted.obs["ident"] = "intervened on"
        adata_test_predicted.obs["cell_stim"] = hold_out_label + "*"

        # Cinema OT works in count space - we compute PCA after prediction for consistency with other methods
        adata_test_predicted.obsm["X_pca"] = adata_inter.uns["pc_transform"].transform(
            adata_test_predicted.X
        )

        return adata_test_predicted
