#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import conceptlab as clab
import numpy as np

entity, project = "andera29", "conceptlab_prime"
df = clab.utils.wandb.download_wandb_project(project, entity)

df["model"].value_counts()

keep_models = ["sccbm", "cinemaot"]

df = df.iloc[np.isin(df["model"].values, keep_models), :]

df.to_csv("../results/sweeps/wandb_cell_cell_sweep.csv")
