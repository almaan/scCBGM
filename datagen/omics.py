from conceptlab.datagen._base import BaseDataGenerator
from conceptlab.utils.types import *
from typing import Dict, Any
import xarray as xr
import numpy as np


def idx_to_mask(idx_vec, size):
    vec = np.zeros(size)
    vec[idx_vec] = 1
    return vec


def categorical(p):
    return np.argmax(np.random.multinomial(1, p))


class OmicsDataGenerator(BaseDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    ):

        rng = np.random.default_rng(seed)

        N, B, T, U, C,F = n_obs, n_batches, n_tissues, n_celltypes, n_concepts, n_vars

        # dirichlet priors
        alpha_U = np.ones(U)  # for cell type
        alpha_C = np.ones(C) / C  # for concepts
        alpha_B = np.ones(B)  # for batches
        alpha_T = np.ones(T)  # for tisse types

        # probability of sampling a batch (to model batch differences)
        p_N = rng.dirichlet(alpha_B)
        # probability of sampling a tissue
        p_Q = rng.dirichlet(alpha_T)
        # probability of sampling a cell type within a batch (to model cell type imbalances)
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


        if zero_inflate:
            # zero inflation probability
            pi = rng.beta(beta_a, beta_b, size=C)

            # add zero inflation
            mask = np.vstack([rng.binomial(1, pi[i], size=F) for i in range(C)])
            cs = cs * mask


        r,s = np.log(baseline_lower),np.log(baseline_upper)
        ws = rng.uniform(r,s, size = F)

        X_mat = np.zeros((N, F))
        C_mat = np.zeros((N, C))
        U_mat = np.zeros(N)
        B_mat = np.zeros(N)
        T_mat = np.zeros(N)



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

        var_names = [f"var_{k}" for k in range(F)]
        obs_names = [f"obs_{k}" for k in range(N)]
        concept_names = [f"concept_{k}" for k in range(C)]
        tissue_names = [f"tissue_{k}" for k in range(T)]
        celltype_names = [f"celltype_{k}" for k in range(U)]
        batch_names = [f"batch_{k}" for k in range(B)]

        T_mat = [f'tissue_{int(x)}' for x in T_mat]
        U_mat = [f'celltype_{int(x)}' for x in U_mat]
        B_mat = [f'batch_{int(x)}' for x in B_mat]

        coords = {
            "obs": obs_names,
            "var": var_names,
            "concept": concept_names,
            "tissue": tissue_names,
            "celltype": celltype_names,
            "batch": batch_names,
        }

        data_vars = dict(
            data=(("obs", "var"), X_mat),
            concepts=(("obs", "concept"), C_mat),
            concept_coef=(("concept", "var"), cs),
            tissues=(("obs", ), T_mat),
            tissue_coef=(("tissue", "var"), tau),
            batches=(("obs", ), B_mat),
            batch_coef=(("batch", "var"), gamma),
            celltypes=(("obs", ), U_mat),
            celltype_coef=(("celltype", "var"), omega),
        )



        dataset = xr.Dataset(data_vars=data_vars,
                             coords=coords)

        return dataset
