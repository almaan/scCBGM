import sys
import os
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)

import torch
import conceptlab as clab
import scanpy as sc

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


# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vae = clab.models.VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae-v2.ckpt"
)
vae.to(device)
vae.eval()


# # z = torch.randn(num_samples, 512)
z = torch.tensor(adata.X.astype(np.float32)).to(device)
# Generate new samples
with torch.no_grad():  # No need to track gradients for generation
    generated_samples = vae.encode(z)

mu, var = generated_samples

plt.figure(figsize=(10, 15))

vae_adata = ad.AnnData(X=mu.cpu().numpy())
vae_adata.obs["batch"] = dataset.batches.to_numpy()
vae_adata.obs["celltype"] = dataset.celltypes.to_numpy()
vae_adata.obs["tissue"] = dataset.tissues.to_numpy()


sc.pp.pca(vae_adata)
sc.pp.neighbors(vae_adata)
sc.tl.umap(vae_adata)
sc.pl.umap(vae_adata, color=["batch", "celltype", "tissue"], ncols=1, title="mu vae")


# Adjust layout to make room for the super title
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("./plotting/" + "vae_mu.png")
plt.show()
plt.clf()


plt.figure(figsize=(10, 15))
vae_adata = ad.AnnData(X=var.cpu().numpy())
vae_adata.obs["batch"] = dataset.batches.to_numpy()
vae_adata.obs["celltype"] = dataset.celltypes.to_numpy()
vae_adata.obs["tissue"] = dataset.tissues.to_numpy()

sc.pp.pca(vae_adata)
sc.pp.neighbors(vae_adata)
sc.tl.umap(vae_adata)
sc.pl.umap(
    vae_adata, color=["batch", "celltype", "tissue"], ncols=1, title="logvar vae"
)


# Adjust layout to make room for the super title
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("./plotting/" + "vae_logvar.png")
plt.show()
