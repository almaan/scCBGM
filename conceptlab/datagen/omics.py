from conceptlab.datagen._base import DataGenerator
from conceptlab.utils.types import *
import conceptlab.utils.constants as _C
from typing import Dict, Any, List
import xarray as xr
import numpy as np

__all__ = ['OmicsDataGenerator']


def idx_to_mask(idx_vec, size):
    vec = np.zeros(size)
    vec[idx_vec] = 1
    return vec


def categorical(p):
    return np.argmax(np.random.multinomial(1, p))


class OmicsDataGenerator(DataGenerator):
    params_key = dict(
        gamma="batch_coef",
        omega="celltype_coef",
        tau="tissue_coef",
        cs="concept_coef",
        ws="baseline",
        std_l="std_libsize",
        p_N="p_batch",
        p_Q="p_tissue",
        p_T="p_celltype_in_tissue",
        p_C="p_concept_in_celltype",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def _model(
        cls,
        n_obs: PositiveInt,
        gamma: np.ndarray,
        omega: np.ndarray,
        tau: np.ndarray,
        cs: np.ndarray,
        ws: np.ndarray,
        p_N: np.ndarray,
        p_Q: np.ndarray,
        p_T: np.ndarray,
        p_C: np.ndarray,
        std_l: np.ndarray,
        std_noise: float = 0.01,
        rng: np.random._generator.Generator | None = None,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """
        Model for generating synthetic omics data.

        Args:
        ---
            n_obs (int): Number of observations.
            gamma (np.ndarray): Batch effects matrix.
            omega (np.ndarray): Cell type-specific effects matrix.
            tau (np.ndarray): Tissue type-specific effects matrix.
            cs (np.ndarray): Concept indicators.
            ws (np.ndarray): Baseline expression levels.
            p_N (np.ndarray): Probabilities for batch ID.
            p_Q (np.ndarray): Probabilities for tissue type.
            p_T (np.ndarray): Probabilities for cell type given tissue type.
            p_C (np.ndarray): Probabilities for concept indicator given cell type.
            std_l (np.ndarray): Standard deviation for library size.
            std_noise (float, optional): Standard deviation for noise. Defaults to 0.01.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.
            seed (int, optional): Seed for random number generator if rng is None. Defaults to 42.

        Returns:
        ---
            Dict[str, np.ndarray]: Dictionary containing generated synthetic data matrices:
                - 'X_mat': Gene expression matrix.
                - 'U_mat': Cell type indicators.
                - 'B_mat': Batch IDs.
                - 'C_mat': Concept indicators.
                - 'T_mat': Tissue type indicators.
        """

        if rng is None:
            rng = np.random.default_rng(seed)

        N, F, C = n_obs, gamma.shape[1], cs.shape[0]

        X_mat = np.zeros((N, F))
        C_mat = np.zeros((N, C))
        U_mat = np.zeros(N)
        B_mat = np.zeros(N)
        T_mat = np.zeros(N)

        # -- Generate Data -- #
        for n in range(N):
            # get batch ID
            b_n = categorical(p_N)
            # get tissue type
            t_n = categorical(p_Q)
            # get cell type
            u_n = categorical(p_T[t_n])
            # get concept indicator
            v_n = rng.binomial(1, p_C[u_n])
            # get library size
            l_n = rng.normal(0, std_l[b_n])
            # get noise
            eps_n = rng.normal(0, std_noise)
            # log lambda
            log_lambda = (
                ws
                + np.sum(cs * v_n[:, None], axis=0)
                + omega[u_n]
                + gamma[b_n]
                + tau[t_n]
                + l_n
                + eps_n
            )
            # sample gene expression
            x_n = rng.poisson(np.exp(log_lambda))

            # register values
            X_mat[n] = x_n
            U_mat[n] = u_n
            B_mat[n] = b_n
            C_mat[n] = v_n
            T_mat[n] = t_n

        return {
            "X_mat": X_mat,
            "U_mat": U_mat,
            "B_mat": B_mat,
            "C_mat": C_mat,
            "T_mat": T_mat,
        }

    @classmethod
    def _build_dataset(
        cls,
        X_mat: np.ndarray,
        U_mat: np.ndarray,
        C_mat: np.ndarray,
        T_mat: np.ndarray,
        B_mat: np.ndarray,
        gamma: np.ndarray,
        omega: np.ndarray,
        tau: np.ndarray,
        ws: np.ndarray,
        cs: np.ndarray,
        std_l: np.ndarray,
        p_N: np.ndarray,
        p_Q: np.ndarray,
        p_T: np.ndarray,
        p_C: np.ndarray,
    ):
        """helper function to build dataset"""

        N, F = X_mat.shape
        B, C, T, U = gamma.shape[0], cs.shape[0], tau.shape[0], omega.shape[0]

        # define names for dimensions
        var_names = [f"var_{k}" for k in range(F)]
        obs_names = [f"obs_{k}" for k in range(N)]
        concept_names = [f"concept_{k}" for k in range(C)]
        tissue_names = [f"tissue_{k}" for k in range(T)]
        celltype_names = [f"celltype_{k}" for k in range(U)]
        batch_names = [f"batch_{k}" for k in range(B)]

        # labelify indicators to match names
        T_mat = [f"tissue_{int(x)}" for x in T_mat]
        U_mat = [f"celltype_{int(x)}" for x in U_mat]
        B_mat = [f"batch_{int(x)}" for x in B_mat]

        # set coordinates for xarray object
        coords = {
            "obs": obs_names,
            "var": var_names,
            "concept": concept_names,
            "tissue": tissue_names,
            "celltype": celltype_names,
            "batch": batch_names,
        }

        # set data variables for xarray object
        data_vars = {
            "data": (("obs", "var"), X_mat),
            _C.DataVars.concept.value: (("obs", "concept"), C_mat),
            cls.params_key["cs"]: (("concept", "var"), cs),
            _C.DataVars.tissue.value: (("obs",), T_mat),
            cls.params_key["tau"]: (("tissue", "var"), tau),
            _C.DataVars.batch.value: (("obs",), B_mat),
            "batch_coef": (("batch", "var"), gamma),
            _C.DataVars.celltype.value: (("obs",), U_mat),
            cls.params_key["omega"]: (("celltype", "var"), omega),
            cls.params_key["std_l"]: (("batch"), std_l),
            cls.params_key["p_N"]: (("batch"), p_N),
            cls.params_key["p_Q"]: (("tissue"), p_Q),
            cls.params_key["p_T"]: (("tissue", "celltype"), p_T),
            cls.params_key["p_C"]: (("celltype", "concept"), p_C),
            cls.params_key["ws"]: (("var"), ws),
        }

        # create xarray dataset
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        return dataset

    @classmethod
    def generate(
        cls,
        n_obs: PositiveInt,
        n_vars: PositiveInt = 1000,
        n_batches: PositiveInt = 3,
        n_tissues: PositiveInt = 2,
        n_celltypes: PositiveInt = 4,
        n_concepts: PositiveInt = 8,
        baseline_lower: NonNegativeFloat = 1,
        baseline_upper: NonNegativeFloat = 5,
        std_batch: NonNegativeFloat = 0.08,
        std_celltype: NonNegativeFloat = 0.08,
        std_tissue: NonNegativeFloat = 0.07,
        std_concept: NonNegativeFloat = 0.05,
        std_libsize_lower: NonNegativeFloat = 0.01,
        std_libsize_upper: NonNegativeFloat = 0.03,
        std_noise: NonNegativeFloat = 0.01,
        beta_a: PositiveFloat = 0.5,
        beta_b: PositiveFloat = 10,
        seed: int = 42,
        zero_inflate: bool = True,
    ) -> xr.Dataset:
        """
        Generate synthetic omics data.

        Args:
            n_obs (int): Number of observations (samples) to generate.
            n_vars (int, optional): Number of variables (features) to generate. Defaults to 1000.
            n_batches (int, optional): Number of batches to simulate. Defaults to 3.
            n_tissues (int, optional): Number of tissue types to simulate. Defaults to 2.
            n_celltypes (int, optional): Number of cell types to simulate. Defaults to 4.
            n_concepts (int, optional): Number of concepts to simulate. Defaults to 8.
            baseline_lower (float, optional): Lower bound for baseline expression levels. Defaults to 1.
            baseline_upper (float, optional): Upper bound for baseline expression levels. Defaults to 5.
            std_batch (float, optional): Standard deviation for batch effects. Defaults to 0.08.
            std_celltype (float, optional): Standard deviation for cell type effects. Defaults to 0.08.
            std_tissue (float, optional): Standard deviation for tissue type effects. Defaults to 0.07.
            std_concept (float, optional): Standard deviation for concept effects. Defaults to 0.05.
            std_libsize_lower (float, optional): Lower bound for standard deviation of library size. Defaults to 0.01.
            std_libsize_upper (float, optional): Upper bound for standard deviation of library size. Defaults to 0.03.
            std_noise (float, optional): Standard deviation for noise. Defaults to 0.01.
            beta_a (float, optional): Alpha parameter for beta distribution in zero inflation. Defaults to 0.5.
            beta_b (float, optional): Beta parameter for beta distribution in zero inflation. Defaults to 10.
            seed (int, optional): Seed for random number generator. Defaults to 42.
            zero_inflate (bool, optional): Whether to add zero inflation to the data. Defaults to True.

        Returns:
            xr.Dataset: An xarray dataset containing the generated synthetic omics data.

        """

        rng = np.random.default_rng(seed)

        N, B, T, U, C, F = n_obs, n_batches, n_tissues, n_celltypes, n_concepts, n_vars

        # dirichlet priors
        alpha_U = np.ones(U)  # for cell type
        alpha_C = np.ones(C) / C  # for concepts
        alpha_B = np.ones(B)  # for batches
        alpha_T = np.ones(T)  # for tisse types

        # probability of sampling a batch (to model batch differences)
        p_N = rng.dirichlet(alpha_B)
        # probability of sampling a tissue
        p_Q = rng.dirichlet(alpha_T)
        # probability of sampling a cell type within a tissue (to model cell type imbalances)
        p_T = rng.dirichlet(alpha_U, size=T)
        # probability of sampling a concept within a cell type
        p_C = rng.dirichlet(alpha_C, size=U)

        # standard deviation for library size
        std_l = rng.uniform(std_libsize_lower, std_libsize_upper, size=B)

        # batch coefficients
        gamma = rng.normal(0, std_batch, size=(B, F))
        # celltype coefficients
        omega = rng.normal(0, std_celltype, size=(U, F))
        # tissue coefficients
        tau = rng.normal(0, std_tissue, size=(T, F))
        # concept coefficients
        cs = rng.normal(0, std_concept, size=(C, F))

        # baseline values
        r, s = np.log(baseline_lower), np.log(baseline_upper)
        ws = rng.uniform(r, s, size=F)

        if zero_inflate:
            # zero inflation probability
            pi = rng.beta(beta_a, beta_b, size=C)

            # add zero inflation
            mask = np.vstack([rng.binomial(1, pi[i], size=F) for i in range(C)])
            cs = cs * mask

        params = dict(
            gamma=gamma,
            omega=omega,
            tau=tau,
            cs=cs,
            ws=ws,
            p_N=p_N,
            p_Q=p_Q,
            p_T=p_T,
            p_C=p_C,
            std_l=std_l,
        )

        data = cls._model(n_obs=N, **params, std_noise=std_noise, rng=rng)

        dataset = cls._build_dataset(**data, **params)

        return dataset

    @classmethod
    def get_params_from_dataset(cls, dataset: xr.Dataset) -> Dict[str, np.ndarray]:
        """helper function to get model parameters from a dataset"""
        params = {key: dataset[val].to_numpy() for key, val in cls.params_key.items()}
        return params

    @classmethod
    def generate_from_dataset(
        cls,
        n_obs: PositiveInt,
        dataset: xr.Dataset,
        new_params: Dict[str, np.ndarray] | None = None,
        std_noise: float = 0.01,
    ) -> xr.Dataset:
        """Generates a new dataset using the same parameters as were used to generte an existing dataset

        Args:
        ----
          n_obs: number of observations
          dataset : dataset to extract parameters from
          new_params: dictionary specifying custom parameters (will overwrite those from the provided dataset)
          std_noise: noise standard deviation (for additional stochasicity)

        Returns:
        ----
          New xarray dataset with n_obs new datapoints

        """

        input_params = cls.get_params_from_dataset(dataset)

        if new_params is not None:
            for key, val in new_params.items():
                input_params[key] = val

        new_data = cls._model(n_obs, **input_params)
        new_dataset = cls._build_dataset(**new_data, **input_params)

        return new_dataset
