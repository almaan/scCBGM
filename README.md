# ConceptLab

This is a toy setup for OMICS data to test different architecture


## Getting started

Install the environment

```
mamba env create -f env.yml
conda activate conceptlab
```

### Running code:

Running experiments using hydra configs

```
python main.py experiment=vae.yaml
```
