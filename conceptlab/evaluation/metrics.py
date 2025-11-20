import numpy as np
from scipy.linalg import sqrtm
import ot


def emd(x, y):
    dist_mat = ot.dist(x, y)
    dist_mat_norm = dist_mat / dist_mat.max()

    emd = ot.emd2([], [], dist_mat_norm, numItermax=int(1e7))
    # emd = np.sum(ot_matrix * dist_mat)

    return emd


def w2(x, y):
    dist_mat = ot.dist(x, y, metric="sqeuclidean")
    w2_sq = ot.emd2([], [], dist_mat, numItermax=int(1e7))
    return np.sqrt(w2_sq)


def sinkhorn_divergence(x, y, reg=0.1):
    dist_mat = ot.dist(x, y, metric="sqeuclidean")
    dist_mat_max = dist_mat.max()
    dist_mat /= dist_mat_max

    sinkhorn_xy = ot.sinkhorn2([], [], dist_mat, reg)

    dist_mat_xx = ot.dist(x, x, metric="sqeuclidean")
    dist_mat_xx_max = dist_mat_xx.max()
    dist_mat_xx /= dist_mat_xx_max
    sinkhorn_xx = ot.sinkhorn2([], [], dist_mat_xx, reg)

    dist_mat_yy = ot.dist(y, y, metric="sqeuclidean")
    dist_mat_yy_max = dist_mat_yy.max()
    dist_mat_yy /= dist_mat_yy_max
    sinkhorn_yy = ot.sinkhorn2([], [], dist_mat_yy, reg)

    sinkhorn_div = (
        dist_mat_max * sinkhorn_xy
        - 0.5 * dist_mat_xx_max * sinkhorn_xx
        - 0.5 * dist_mat_yy_max * sinkhorn_yy
    )

    return sinkhorn_div


def rbf_kernel(x, y, sigma=1.0):
    """Compute the RBF kernel between x and y."""
    x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y**2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-dist / (2 * sigma**2))


def normalize_kernel_matrix(K):
    """Normalize a kernel matrix K."""
    diag = np.sqrt(np.diag(K))
    return K / np.outer(diag, diag)


def polynomial_kernel(x, y, degree=3, coef=1.0):
    from sklearn.metrics.pairwise import polynomial_kernel as kernel

    return kernel(x.astype(np.float64), y.astype(np.float64), coef0=coef, degree=degree)


def kernel_inception_distance(real_features, gen_features, degree=3, coef=1.0):
    """
    Compute Kernel Inception Distance (KID) using a given kernel.

    Args:
        real_features (np.ndarray): Features from real data, shape (m, d)
        gen_features (np.ndarray): Features from generated data, shape (n, d)
        degree (int): Degree of the polynomial kernel.
        coef (float): Bias term in the kernel.

    Returns:
        float: KID score (squared MMD).
    """
    m, n = real_features.shape[0], gen_features.shape[0]

    # Compute polynomial kernels
    K_xx = polynomial_kernel(real_features, real_features, degree, coef)  # Real-Real
    K_yy = polynomial_kernel(gen_features, gen_features, degree, coef)  # Gen-Gen
    K_xy = polynomial_kernel(real_features, gen_features, degree, coef)  # Real-Gen

    # Exclude diagonal elements for K_xx and K_yy
    K_xx_sum = (np.sum(K_xx) - np.sum(np.diag(K_xx))) / (m * (m - 1))
    K_yy_sum = (np.sum(K_yy) - np.sum(np.diag(K_yy))) / (n * (n - 1))
    K_xy_sum = np.sum(K_xy) / (m * n)

    # Compute squared MMD (KID)
    kid_score = K_xx_sum + K_yy_sum - 2 * K_xy_sum
    return kid_score


def frechet_distance(X, Y):
    """
    Compute the Fréchet Distance between two datasets X and Y.
    X: m x d matrix (m observations, d features)
    Y: n x d matrix (n observations, d features)
    """
    # Step 1: Compute the means
    mu_X = np.mean(X, axis=0)  # Mean vector of X
    mu_Y = np.mean(Y, axis=0)  # Mean vector of Y

    # Step 2: Compute the covariance matrices
    cov_X = np.cov(X, rowvar=False)  # Covariance matrix of X
    cov_Y = np.cov(Y, rowvar=False)  # Covariance matrix of Y

    # Step 3: Compute the squared Euclidean distance between the means
    mean_diff = np.sum((mu_X - mu_Y) ** 2)

    # Step 4: Compute the matrix square root of (cov_X @ cov_Y)
    cov_prod_sqrt = sqrtm(cov_X @ cov_Y)

    # Handle numerical precision issues (complex numbers with small imaginary parts)
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real

    # Step 5: Compute the trace part of the formula
    trace_term = np.trace(cov_X + cov_Y - 2 * cov_prod_sqrt)

    # Step 6: Combine terms to get the Fréchet Distance
    frechet_dist = mean_diff + trace_term
    return frechet_dist


def mmd(X, Y, sigma=1.0, kernel=rbf_kernel, kernel_params={}):
    """
    Compute the unbiased MMD^2 between two matrices X and Y.
    X: (m, d) matrix
    Y: (n, d) matrix
    sigma: kernel bandwidth parameter
    """
    m, n = X.shape[0], Y.shape[0]

    K_XX = kernel(X, X, **kernel_params)
    K_YY = kernel(Y, Y, **kernel_params)
    K_XY = kernel(X, Y, **kernel_params)

    # Remove diagonal elements for unbiased estimation
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    # Compute MMD^2
    term_xx = K_XX.sum() / (m * (m - 1))  # Mean of K_XX without diagonal
    term_yy = K_YY.sum() / (n * (n - 1))  # Mean of K_YY without diagonal
    term_xy = K_XY.sum() / (m * n)  # Mean of K_XY

    mmd_squared = term_xx + term_yy - 2 * term_xy

    return mmd_squared


def cosine_sim(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the magnitude (L2 norm) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculate the cosine similarity
    # The calculation is robust against division by zero
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def _calculate_median_distance(X, Y):
    """
    Calculates the median of pairwise distances between points in X and Y.
    This is used for the median heuristic for setting the RBF kernel gamma.
    """
    # Combine the two point clouds
    Z = np.concatenate([X, Y], 0)

    # Calculate pairwise squared Euclidean distances
    Z_norm = np.sum(Z**2, axis=1, keepdims=True)
    dist_sq = Z_norm - 2 * np.dot(Z, Z.T) + Z_norm.T

    # Get the unique, non-zero distances and take the square root
    distances = np.sqrt(np.clip(dist_sq, a_min=0, a_max=None))

    # Get the upper triangular part to avoid duplicates and diagonal zeros
    triu_indices = np.triu_indices_from(distances, k=1)
    unique_distances = distances[triu_indices]

    # Return the median
    return np.median(unique_distances)


def rbf_kernel(X, Y, gamma=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of vectors.
    The RBF kernel is defined as: k(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        X (np.ndarray): An array of shape (m, d), where m is the number of points
                        and d is the dimension of each point.
        Y (np.ndarray): An array of shape (n, d), where n is the number of points
                        and d is the dimension of each point.
        gamma (float): The bandwidth parameter of the RBF kernel. It controls
                       the "width" of the kernel.

    Returns:
        np.ndarray: The kernel matrix of shape (m, n).
    """
    # Calculate the pairwise squared Euclidean distances using broadcasting
    # ||x - y||^2 = ||x||^2 - 2 * x^T * y + ||y||^2
    X_norm = np.sum(X**2, axis=1, keepdims=True)
    Y_norm = np.sum(Y**2, axis=1, keepdims=True)

    XY = np.dot(X, Y.T)

    dist_sq = X_norm - 2 * XY + Y_norm.T

    # Apply the RBF kernel function
    K = np.exp(-gamma * dist_sq)
    return K


def calculate_mmd(X, Y, gamma=None):
    """
    Calculates the Maximum Mean Discrepancy (MMD) using the unbiased estimator.
    This function computes the squared MMD between two point clouds.

    Args:
        X (np.ndarray): The first point cloud, an array of shape (m, d).
                        In this case, d=128.
        Y (np.ndarray): The second point cloud, an array of shape (n, d).
                        In this case, d=128.
        gamma (float, optional): The bandwidth for the RBF kernel. If None, it is
                                 set using the median heuristic, which is a robust
                                 way to account for data dimensionality and scale.
                                 Defaults to None.

    Returns:
        float: The squared MMD value.
    """
    m, d = X.shape
    n = Y.shape[0]

    if m < 2 or n < 2:
        # The unbiased estimator is not defined for fewer than 2 samples.
        return 0.0

    # If gamma is not specified, use the median heuristic. This is crucial for
    # high-dimensional data. gamma = 1 / (2 * sigma^2), where sigma is the median distance.
    if gamma is None:
        median_dist = _calculate_median_distance(X, Y)
        # Avoid division by zero if all points are identical
        if median_dist > 0:
            gamma = 1.0 / (2 * median_dist**2)
        else:
            gamma = 1.0 / d  # Fallback heuristic

        # print("Using median heuristic for gamma:", gamma)
    # else:
    #     print("Using provided gamma:", gamma)
    # Calculate the kernel matrices
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    # Unbiased MMD^2 estimator.
    # We set the diagonal elements to 0 to avoid comparing a point with itself.
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    term1 = np.sum(K_XX) / (m * (m - 1))
    term2 = np.sum(K_YY) / (n * (n - 1))
    term3 = -2 * np.mean(K_XY)

    mmd_sq = term1 + term2 + term3

    # MMD can sometimes be slightly negative due to numerical precision, so we clamp at 0.
    return np.clip(mmd_sq, a_min=0, a_max=None)
