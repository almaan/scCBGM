import conceptlab as clab
import numpy as np
import pandas as pd
import anndata as ad


from tap import Tap


class ArgParser(Tap):
    out_dir: str = "."
    filename: str = "synthetic.h5ad"
    n_obs: int = 25000
    n_vars: int = 5000
    n_celltypes: int = 6
    n_tissues: int = 3
    n_batches: int = 2
    n_concepts: int = 5
    baseline_lower: float = 1
    baseline_upper: float = 5
    std_batch: float = 0.08
    std_celltype: float = 0.08
    std_tissue: float = 0.07
    std_concept: float = 0.05
    std_libsize_lower: float = 0.01
    std_libsize_upper: float = 0.03
    std_noise: float = 0.01
    cov_prior_range: float = 5
    beta_a: float = 1
    beta_b: float = 0.5
    seed: int = 42
    zero_inflate: bool = True
    use_concept_dependency: bool = False


def main():

    args = ArgParser().parse_args()

    print("Generate base dataset - this might take a while... :hourglass:")
    dataset = clab.datagen.omics.OmicsDataGenerator.generate(**args.as_dict())

    X = dataset.data.to_pandas()
    C = dataset.concepts.to_pandas()

    Xs = []
    Cs = []
    split_cols = []

    print("Start Generating counterfactuals... :zap:")

    for concept_name in C.columns:

        print("Rendering counterfactuals for concept: {}".format(concept_name))

        is_off = np.where(C[concept_name].values == 0)[0]
        is_active = np.where(C[concept_name].values == 1)[0]

        if (sum(is_active) == 0) or (sum(is_off) < 2):
            continue

        np.random.shuffle(is_off)
        mid_ix = int(0.5 * len(is_off))
        train_ix, test_ix = is_off[0:mid_ix], is_off[mid_ix:]

        split_labels = np.array(["train"] * X.shape[0], dtype="<U64")

        split_labels[test_ix] = "modify"

        new_concepts = C.copy()
        new_concepts.loc[:, concept_name] = 1

        X_new = clab.datagen.omics.OmicsDataGenerator.generate_intervention(
            dataset, new_concepts
        )
        X_new = X_new.iloc[test_ix].copy()
        C_new = new_concepts.iloc[test_ix].copy()

        split_col_name = concept_name + "_split"
        split_cols.append(split_col_name)
        X_new[split_col_name] = "hold_out"
        X[split_col_name] = split_labels

        Xs.append(X_new)
        Cs.append(C_new)

    print("Merging and formatting generated data :link:")

    Xs_df = pd.concat(Xs, axis=0)
    Xs_df.fillna("drop", inplace=True)
    Xs_df = pd.concat([X, Xs_df])

    Cs_df = pd.concat(Cs, axis=0)
    Cs_df = pd.concat([C, Cs_df])

    obs = Xs_df[split_cols].copy()
    Xs_df.drop(columns=split_cols, inplace=True)

    obs["original_index"] = obs.index
    obs.index = ["obs_{}".format(x) for x in range(len(obs))]

    Xs_df.index = obs.index
    Cs_df.index = obs.index

    Print("Creating anndata")
    adata = ad.AnnData(Xs_df, obs=obs, var=pd.DataFrame([], index=Xs_df.columns))
    adata.obsm["concepts"] = Cs_df

    filename = (
        args.filename if args.filename.endswith(".h5ad") else args.filename + ".h5ad"
    )

    print("Saved synthetic dataset to {}/{}".format(args.out_dir, filename))

    adata.write_h5ad("{}/{}".format(args.out_dir, filename))


if __name__ == "__main__":
    main()
