import sys
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)

import torch
import conceptlab as clab
import scanpy as sc
import xarray as xr

print("Generating OmicsData")


n_vars = 34455
num_samples = 10000

# n_tissues = 3
# n_celltypes = 10
# n_batches = 2
# n_concepts = 8
# dataset = clab.datagen.omics.OmicsDataGenerator.generate(n_obs = num_samples,
#                                                          n_vars = n_vars,
#                                                          n_tissues=n_tissues,
#                                                          n_celltypes=n_celltypes,
#                                                          n_batches = n_batches,
#                                                          n_concepts = n_concepts,
#                                                         )


dataset = xr.open_dataset("./data/complete_set_dataset.nc")

dataset = clab.datagen.omics.OmicsDataGenerator.generate_from_dataset(
    n_obs=num_samples, dataset=dataset
)

print(dataset)
B_mat = dataset.batch
U_mat = dataset.celltype
numpy_array = dataset.data.to_numpy()

# # # B_mat=dataset.batch
adata = ad.AnnData(X=numpy_array)
adata.obs["batch"] = dataset.batches.to_numpy()
adata.obs["celltype"] = dataset.celltypes.to_numpy()
adata.obs["tissue"] = dataset.tissues.to_numpy()

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)


df1 = pd.DataFrame(dataset.data.to_numpy())
df1["source"] = "omics"


# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.tensor(adata.X.astype(np.float32)).to(device)


print("Generating vae")
vae = clab.models.VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae-v3.ckpt"
)
vae.to(device)
vae.eval()


# Generate new samples
with torch.no_grad():  # No need to track gradients for generation
    mu, logvar = vae.encode(x)
    z = vae.reparameterize(mu, logvar)
    generated_samples = vae.decode(z)

generated_samples = generated_samples.cpu()

df2 = pd.DataFrame(generated_samples)
df2["source"] = "vae"


print("Generating vae_cbm")
vae = clab.models.CB_VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae_cbm-v3.ckpt"
)
vae.to(device)
vae.eval()


# Generate new samples
with torch.no_grad():  # No need to track gradients for generation
    mu, logvar = vae.encode(x)
    z = vae.reparameterize(mu, logvar)
    generated_samples = vae.decode(z)

generated_samples = generated_samples.cpu()

df3 = pd.DataFrame(generated_samples)
df3["source"] = "vae_cbm"


X_df = pd.concat([df1, df2, df3], ignore_index=True)

df_subset = X_df.drop("source", axis=1)

adata = ad.AnnData(X=df_subset.values)
adata.obs["source"] = X_df["source"].values


# vae_adata = ad.AnnData(X=mu.cpu().numpy())
# vae_adata.obs['batch'] = dataset.batches.to_numpy()
# vae_adata.obs['celltype'] = dataset.celltypes.to_numpy()
# vae_adata.obs['tissue'] = dataset.tissues.to_numpy()


# print("normalizing")
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["source"], title="omics generated")

plt.savefig("./plotting/" + "omics_vs_modeled_using_latenet.png")
plt.show()
plt.clf()
