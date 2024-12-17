import numpy as np
from scipy.linalg import sqrtm


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


def mmd(X, Y, sigma=1.0, kernel=polynomial_kernel, kernel_params={}):
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
