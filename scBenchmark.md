# Running the single cell benchmarks (Kang and Cui)

All the configs are in `fm_config` and wandb sweeps are in `fm_config/sweep`

The main script for launching all experiments is `scripts/benchmark_scdata.py`

The two main required components of the config are a model and a dataset.

Right now we have Kang and Cui Datasets configs.

Each model should have a `train()` and a `predict_intervention()` function.

See an example in `conceptlab/models/cb_fm.py` with the class `CBMFM_MetaTrainer`


## Launching a sweep:

Go in `/jobs` and run:

`./init_sweep.sh ../fm_config/sweeps/kang_cbm.yaml WANDB_PROJECT WANDB_ENTITY`

If you're not using `uv` but a standard conda/mamba environment then add a --conda_mode flag, i.e.,:

`./init_sweep.sh ../fm_config/sweeps/kang_cbm.yaml WANDB_PROJECT WANDB_ENTITY -c`

Where you can feed the path to the sweep you want to run.

The output should look like:

```
Launching sweep from config: ../fm_config/sweeps/kang_cbmfm_raw.yaml
Project: conceptlab | Entity: USER
✅ Created sweep: USER/conceptlab/idqe1p2q
```

Copy paste the sweep path in `pcluster_sweep.sh` and run it !

`sbatch --export=ALL,ROOT_DIR='path_to_conceptlab_root_dir' pclsuter_sweep.sh SWEEP_ID`

If you're not using `uv` but a standard conda/mamba environment then add a --conda_mode flag

`sbatch --export=ALL,ROOT_DIR='path_to_conceptlab_root_dir' pclsuter_sweep.sh -c SWEEP_ID`

Check your results in wandb !

## Results aggregation and plots

An example of results aggregation (that pulls sweeps from wandb and plots them) is given in `/notebooks/analysis/results_aggregation.ipynb`
