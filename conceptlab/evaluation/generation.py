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


def mse_loss(gt, pred, raw_data=None, plot=True, normalize=False):

    x_pred = _normalize(gt, normalize)
    x_true = _normalize(pred, normalize)

    mse = np.mean((x_true - x_pred) ** 2)

    if raw_data is not None:
        noise_levels = np.arange(0, 1.1, 0.1)
        mse_noise_array = []
        for noise_level in noise_levels:

            noisy_gex = _multinomial_resampling(raw_data, noise_level)
            noisy_gex = noisy_gex / noisy_gex.sum(axis=1, keepdims=True) * 1e4
            noisy_gex = np.log1p(noisy_gex)
            mse_noise = np.mean((x_true - noisy_gex) ** 2)
            mse_noise_array.append(mse_noise)

        if plot:
            # Create the MSE vs Noise Level plot
            fig = go.Figure()

            # Plot the MSE values against noise levels
            fig.add_trace(
                go.Scatter(
                    x=noise_levels,
                    y=mse_noise_array,
                    mode="lines+markers",
                    name="Noise MSE",
                )
            )
            # Add a horizontal line to represent the test MSE
            fig.add_trace(
                go.Scatter(
                    x=[noise_levels[0], noise_levels[-1]],
                    y=[mse, mse],
                    mode="lines",
                    name="Test MSE",
                    line=dict(color="red"),
                )
            )
            # Customize the layout
            fig.update_layout(
                title="MSE vs Noise Level",
                xaxis_title="Noise Level",
                yaxis_title="MSE",
                width=300,
                height=600,
            )

            # Log the plot to Weights & Biases
            wandb.log({"mse_vs_noise_level": fig})

    else:
        mse_noise = None

    return mse
