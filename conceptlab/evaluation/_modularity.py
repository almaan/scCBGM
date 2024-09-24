import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import community

def compute_modularity(data, labels, n_neighbors = 30):

    # Step 1: Compute nearest neighbors using sklearn's NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Step 2: Build a sparse adjacency matrix from neighbors
    n_samples = data.shape[0]

    # Create row, col, and data arrays for constructing the sparse matrix
    rows = np.repeat(np.arange(n_samples), indices.shape[1])
    cols = indices.flatten()
    data = np.ones_like(rows)

    # Use scipy's csr_matrix to create a sparse adjacency matrix
    adj_matrix_sparse = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    G = nx.from_scipy_sparse_array(adj_matrix_sparse)

    attrs = {k:l for k,l in enumerate(labels)}

    nx.set_node_attributes(G, attrs, 'labels')

    partition = {node: G.nodes[node]['labels'] for node in G.nodes()}

    # Compute the modularity score
    modularity = community.modularity(partition, G)

    return modularity



