#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import conceptlab as clab
import numpy as np

entity, project = "rb-aiml", "clab_performance"
df = clab.utils.wandb.download_wandb_project(project, entity)

df["model"].value_counts()

keep_models = ["skip_cb_vae", "cem_vae", "vae", "cvae", "cb_vae", "biolord"]

df = df.iloc[np.isin(df["model"].values, keep_models), :]

df.to_csv("../results/sweeps/wandb_performance_sweep.csv")
