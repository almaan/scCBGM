import numpy as np
import plotly.graph_objects as go
import wandb


def _multinomial_resampling(X: np.ndarray, noise_level: float = 0):

    ns = X.sum(axis=1)
    ps = X.astype(np.float64) / ns.reshape(-1, 1)
    n_obs, n_var = X.shape

    if noise_level < 0:
        print(
            "[WARNING] : The noise level for resampling has to be in [0,1]. You chose {} - adjusting value to 0".format(
                noise_level
            )
        )
        noise_level = 0

    if noise_level > 1:
        print(
            "[WARNING] : The noise level for resampling has to be in [0,1]. You chose {} - adjusting value to 1".format(
                noise_level
            )
        )
        noise_level = 1

    if noise_level > 0:
        alpha = np.ones(n_var)
        eps = np.random.dirichlet(alpha, size=n_obs)
        ps = (1 - noise_level) * ps + noise_level * eps
        ps = ps / ps.sum(axis=1, keepdims=True)

    X_new = np.vstack(
        [np.random.multinomial(ns[i], pvals=ps[i]) for i in range(len(ns))]
    )

    return X_new


def _normalize(x, normalize=False):
    if normalize:
        return np.log1p(x / (x.sum(axis=1, keepdims=True) + 1e-8) * 1e4)
    return x


def r2_score(x_true, x_pred):
    from sklearn.metrics import r2_score

    return r2_score(x_true, x_pred)


def mse_loss(
    x_true,
    x_pred,
):

    mse = np.mean((x_true - x_pred) ** 2)

    return mse


def cosine_similarity(x_true, x_pred):
    from numpy.linalg import norm

    rowwise_cos = np.sum(x_true * x_pred, axis=1) / (
        norm(x_true, axis=1) * norm(x_pred, axis=1)
    )
    cos_sim = np.mean(rowwise_cos)

    return cos_sim


def rowwise_correlation(x_true, x_pred):
    """
    Compute the average row-wise Pearson correlation
    between x_true and x_pred, vectorized.

    Args:
        x_true (ndarray): shape (n_samples, n_features)
        x_pred (ndarray): shape (n_samples, n_features)

    Returns:
        float: average correlation across rows
    """
    x_true = np.asarray(x_true, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)

    # Center each row (subtract row mean)
    x_true_centered = x_true - x_true.mean(axis=1, keepdims=True)
    x_pred_centered = x_pred - x_pred.mean(axis=1, keepdims=True)

    # Row-wise numerator (dot product)
    num = np.sum(x_true_centered * x_pred_centered, axis=1)

    # Row-wise denominator (product of norms)
    denom = np.linalg.norm(x_true_centered, axis=1) * np.linalg.norm(
        x_pred_centered, axis=1
    )

    # Avoid division by zero
    corrs = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)

    return np.mean(corrs)
