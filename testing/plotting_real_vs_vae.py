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

print("Reading data")
data_pth = "./data/subset_cells_50000.h5ad"
data = ad.read_h5ad(data_pth)

data = data.to_df().values
subset_cells = data[0:10000]


df1 = pd.DataFrame(subset_cells)
df1["source"] = "genexcell"
num_samples, n_var = subset_cells.shape


# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Generating data")
vae = clab.models.VAE.load_from_checkpoint(
    "/data/bucket/ismaia11/conceptlab/models/vae_real.ckpt"
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


print(df2)
X_df = pd.concat([df1, df2], ignore_index=True)

df_subset = X_df.drop("source", axis=1)

adata = ad.AnnData(X=df_subset.values)
adata.obs["source"] = X_df["source"].values

# print(adata.X)
# print("normalizing")
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["source"], title="real vs generated")

plt.savefig("./plotting/" + "vae_vs_real.png")
plt.show()
plt.clf()
