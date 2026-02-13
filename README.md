# 🌱 scCBGM

Repository for `scCBGM` method and accompanying evaluation presented in the manuscript "scCBGM: Single-cell Editing via Concept Bottlenecks".

In short, scCBGM is a method for that leverages an abduction-action-prediction
counterfactual generation of scRNA-seq data using a combination of concept
bottleneck generative models.


# 📦 conceptlab Package

The `conceptlab` package is a comprehensive suite used to produce the results in
the associated manuscript. While it supports complex workflows — including
synthetic data generation, hyperparameter sweeps, and benchmarking of different methods —it also
offers a streamlined, `scanpy`-inspired API for our core model: **scCBGM**.

:alert: If you _are_ looking to use the `scCBGM` model only to analyze your own data, the API is what you're looking for. 

### Installation

Set up the environment (we recommend using `micromamba`, but you can use any package and environment manager):

```sh
micromamba env create -f env.yaml
micromamba activate conceptlab
```

Install the package:

```sh
pip install
```

## 🚀 Quick Start: The scCBGM API

The `scCBGM` API is built specifically for `anndata` users, taking care of all
the annoying stuff (like setting up training loops and formatting or batching
data).

### Basic Workflow

```python

import conceptlab.api as cb
import scanpy as sc

# 1. Setup Data
adata = sc.read_h5ad('path_to_data.h5ad')

# Use our default normalization or your own preferred strategy
cb.pp.norm.default_normalization(adata)

# 2. Curate Concepts
# Supports both categorical and numerical columns in adata.obs
concept_cols = ['stimulation', 'day'] 
cb.pp.concepts.curate_concepts(adata, concept_cols=concept_cols, concept_key='concepts')

# 3. Model Training
model = cb.models.scCBGM()
model.fit(adata, num_epochs=100, concept_key='concepts')

# 4. Inference & Reconstruction
# Returns known and unknown concept embeddings (use inplace=True to add to adata)
known, unknown = model.encode(adata_test) 
reconstructed = model.reconstruct(adata_test)

# 5. Counterfactual Intervention
# Define the shift (e.g., changing 'day' from 3 to 9)
new_concepts = cb.pp.concepts.intervene_on_concepts(
    adata, 
    concept_key='concepts', 
    conditions={'day': {'3': '9'}}
)

# Generate the counterfactual state
counterfactual_data = model.intervene(adata, new_concepts, concept_key='concepts')

# 6. Save & Load
model.save('model.pt')
model = cb.models.scCBGM.load('model.pt')
```

Most operations (e.g., `decode`, `encode` , `infer`) have `inplace` options,
this will update your `anndata` inplace (adding new layers for reconstruction
and `.obsm` views for concepts).


### 📦 Complete suite Suite 

The complete suite of functionality encompass several different modules that all serve a specific purpose; the modules are:

- `datagen` → generation of synthetic data (for evaluation purposes)
- `data` → dataloaders
- `evaluation` → evaluation functions
- `models` → model classes, including our `scCBGM` model and baselines
- `preprocess` → preprocessing tools
- `api` - detailed above

Examples of usage can be found in `analysis/notebooks` as well as `main.py`.

## Reproducing Results
For more information on how to reproduce the results presented in the associated
manuscript, see of [reproducibility](guides/reproduce.md) page. This leverages the larger suite of modules.


