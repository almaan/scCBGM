#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import conceptlab as clab
import numpy as np

entity, project = "rb-aiml", "clab_kang"
df = clab.utils.wandb.download_wandb_project(project, entity)

df.to_csv("../results/sweeps/wandb_kang_results.csv")
