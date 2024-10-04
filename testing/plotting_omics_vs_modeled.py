import sys
import os
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)

import torch
import conceptlab as clab
import scanpy as sc

print("Generating OmicsData")


n_vars = 34455
num_samples = 10000

n_tissues = 3
n_celltypes = 10
n_batches = 2
n_concepts = 8
dataset = clab.datagen.omics.OmicsDataGenerator.generate(
    n_obs=num_samples,
    n_vars=n_vars,
    n_tissues=n_tissues,
    n_celltypes=n_celltypes,
    n_batches=n_batches,
    n_concepts=n_concepts,
)


df1 = pd.DataFrame(dataset.data.to_numpy())
df1["source"] = "genexcell"


# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Generating vae")
vae = clab.models.VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae-v2.ckpt"
)
vae.to(device)
vae.eval()


z = torch.randn(num_samples, 512).to(device)
# Generate new samples
with torch.no_grad():  # No need to track gradients for generation
    generated_samples = vae.decode(z)

generated_samples = generated_samples.cpu()

df2 = pd.DataFrame(generated_samples)
df2["source"] = "vae"


print("Generating vae_cbm")
vae = clab.models.CB_VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae_cbm-v2.ckpt"
)
vae.to(device)
vae.eval()


z = torch.randn(num_samples, 512).to(device)
# Generate new samples
with torch.no_grad():  # No need to track gradients for generation
    generated_samples = vae.decode(z)

generated_samples = generated_samples.cpu()

df3 = pd.DataFrame(generated_samples)
df3["source"] = "vae_cbm"


X_df = pd.concat([df1, df2, df3], ignore_index=True)

df_subset = X_df.drop("source", axis=1)

adata = ad.AnnData(X=df_subset.values)
adata.obs["source"] = X_df["source"].values

# print("normalizing")
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["source"], title="omics generated")

plt.savefig("./plotting/" + "omics_vs_modeled.png")
plt.show()
plt.clf()
