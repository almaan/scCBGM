#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


# In[8]:


import pandas as pd
import conceptlab as clab
import numpy as np


# In[3]:


entity, project = "rb-aiml", "clab_hparam_sweep"
df = clab.utils.wandb.download_wandb_project(project, entity)


# In[6]:


df["model"].value_counts()


# In[7]:


keep_models = ["skip_cb_vae", "cem_vae", "vae", "cvae", "cb_vae"]


# In[9]:


df = df.iloc[np.isin(df["model"].values, keep_models), :]


# In[10]:


df.to_csv("../results/wandb_hparam_sweep.csv")


# In[ ]:
