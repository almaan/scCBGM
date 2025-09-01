# Running the single cell benchmarks (Kang and Cui)

All the configs are in `fm_config` and wandb sweeps are in `fm_config/sweep`

The main script for launching all experiments is `scripts/benchmark_scdata.py`

Basically it needs a model and a dataset.

Each model should have a `train()` and a `predict_intervention()` function.


See an example in `conceptlab/models/cb_fm.py` with the class `CBMFM_MetaTrainer`

