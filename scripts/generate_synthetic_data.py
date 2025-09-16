import conceptlab as clab
import numpy as np
import pandas as pd
import anndata as ad
import itertools
import random


from tap import Tap


class ArgParser(Tap):
    out_dir: str = "."
    filename: str = "synthetic.h5ad"
    n_hold_out: int = 5
    n_obs: int = 10000
    n_vars: int = 5000
    n_celltypes: int = 4
    n_tissues: int = 3
    n_batches: int = 2
    n_concepts: int = 5
    baseline_lower: float = 1
    baseline_upper: float = 5
    std_batch: float = 0.06
    std_celltype: float = 0.04
    std_tissue: float = 0.05
    std_concept: float = 0.50
    std_libsize_lower: float = 0.01
    std_libsize_upper: float = 0.03
    std_noise: float = 0.01
    cov_prior_range: float = 5
    beta_a: float = 1
    beta_b: float = 0.8
    seed: int = 42
    zero_inflate: bool = True
    use_concept_dependency: bool = False


def main():

    args = ArgParser().parse_args()

    print("Generate base dataset - this might take a while... :hourglass:")
    dataset = clab.datagen.omics.OmicsDataGenerator.generate(**args.as_dict())

    X = dataset.data.to_pandas()
    C = dataset.concepts.to_pandas()
    U = dataset.celltypes.to_pandas()

    U = pd.DataFrame(U, columns=["cell_type"])

    ct_names = U["cell_type"].unique().tolist()
    concept_names = C.columns.tolist()

    combinations = list(itertools.product(ct_names, concept_names))
    random.shuffle(combinations)
    sel_combinations = combinations[: args.n_hold_out]

    Xs = []
    Cs = []
    Us = []
    split_cols = []

    print("Start Generating counterfactuals... :zap:")

    for intervention_id, (ct_name, concept_name) in enumerate(sel_combinations):

        print(
            "Rendering counterfactuals for combination (cell type, concept ) : ({},{}) ".format(
                ct_name, concept_name
            )
        )

        is_ct = U["cell_type"].values == ct_name
        is_on = C[concept_name].values == 1

        # cells that are ct and concept on - we will drop these
        drop_ix = np.where(is_ct & (is_on))[0]
        # cells that are ct and concept off
        ct_off_ix = np.where(is_ct & (~is_on))[0]

        np.random.shuffle(ct_off_ix)

        # half of the "off cells" are used for modification, the other half remain in training
        mid_point = int(len(ct_off_ix) / 2)
        modify_ix = ct_off_ix[:mid_point]

        split_labels = np.array(["train"] * X.shape[0], dtype="<U64")
        split_labels[modify_ix] = "modify"
        split_labels[drop_ix] = "drop"

        new_concepts = C.copy()
        new_concepts.loc[:, concept_name] = 1

        X_new = clab.datagen.omics.OmicsDataGenerator.generate_intervention(
            dataset, new_concepts
        )
        X_new = X_new.iloc[modify_ix].copy()
        C_new = new_concepts.iloc[modify_ix].copy()
        U_new = U.iloc[modify_ix].copy()

        split_col_name = "intervention_{}".format(intervention_id)
        split_cols.append(split_col_name)
        X_new[split_col_name] = "hold_out"
        X[split_col_name] = split_labels

        Xs.append(X_new)
        Cs.append(C_new)
        Us.append(U_new)

    print("Merging and formatting generated data :link:")

    Xs_df = pd.concat(Xs, axis=0)
    Xs_df.fillna("drop", inplace=True)
    Xs_df = pd.concat([X, Xs_df])

    Cs_df = pd.concat(Cs, axis=0)
    Cs_df = pd.concat([C, Cs_df], axis=0)

    Us_df = pd.concat(Us, axis=0)
    Us_df = pd.DataFrame(pd.concat([U, Us_df], axis=0))

    obs = Xs_df[split_cols].copy()
    obs = pd.concat([obs, Us_df], axis=1)

    Xs_df.drop(columns=split_cols, inplace=True)

    obs["original_index"] = obs.index
    obs.index = ["obs_{}".format(x) for x in range(len(obs))]

    Xs_df.index = obs.index
    Cs_df.index = obs.index

    print("Creating anndata")
    adata = ad.AnnData(Xs_df, obs=obs, var=pd.DataFrame([], index=Xs_df.columns))
    adata.obsm["concepts"] = Cs_df

    # TODO: I'M REALLY STUPID - FIX THIS

    for k, (_, concept_name) in enumerate(sel_combinations):
        obsm_name = "intervention_{}_concept".format(k)
        adata.obsm[obsm_name] = adata.obsm["concepts"].copy()
        adata.obsm[obsm_name].rename(
            columns={concept_name: "target_concept"}, inplace=True
        )

    adata.uns["intervention_information"] = pd.DataFrame(
        sel_combinations, columns=["cell_type", "concept"]
    )

    filename = (
        args.filename if args.filename.endswith(".h5ad") else args.filename + ".h5ad"
    )

    print("Saved synthetic dataset to {}/{}".format(args.out_dir, filename))

    adata.write_h5ad("{}/{}".format(args.out_dir, filename))


if __name__ == "__main__":
    main()
