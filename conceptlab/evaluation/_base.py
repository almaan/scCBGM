from abc import ABCMeta, abstractmethod, abstractclassmethod
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Literal, Tuple
import numpy as np
from conceptlab.utils import helpers


class EvaluationClass(ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def score(cls, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractclassmethod
    def save(cls, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def _score_fun(cls, X_new, X_old, **kwargs):
        pass

    @classmethod
    def score(
        cls,
        X_old: pd.DataFrame,
        X_new: pd.DataFrame,
        concepts_old: pd.DataFrame,
        concepts_new: pd.DataFrame,
        concept_coefs: pd.DataFrame,
        use_neutral: bool = True,
        invert_neg: bool = False,
    ) -> Tuple[Dict[str, Any]]:

        concept_names = concepts_old.columns
        concept_uni_vars = (concept_coefs != 0).sum(axis=0)
        concept_neutral_vars = concept_coefs.columns[concept_uni_vars.values == 0]

        direction_to_true = dict(
            neu=0,
            pos=1,
            neg=1,
        )

        results = dict()
        curves = dict()

        for concept_name in concept_names:

            results[concept_name] = dict()
            curves[concept_name] = dict()

            concept_delta = (
                concepts_new.loc[:, concept_name] - concepts_old.loc[:, concept_name]
            ).values

            concept_delta_mean = concept_delta.mean()

            if concept_delta_mean == 0:
                continue

            pos_delta = concept_delta > 0
            neg_delta = concept_delta < 0

            results[concept_name] = dict()

            var_names_dict = dict()

            coefs_k = concept_coefs.loc[concept_name, :]

            var_names_dict["pos"] = coefs_k.index[coefs_k.values > 0]
            var_names_dict["neg"] = coefs_k.index[coefs_k.values < 0]

            if use_neutral:
                var_names_dict["neu"] = concept_neutral_vars
            else:
                var_names_dict["neu"] = coefs_k.index[coefs_k.values == 0]

            vals_new = X_new.values
            vals_old = X_old.values

            for delta, dropped, ivn in zip(
                [pos_delta, neg_delta], [False, True], ["0->1", "1->0"]
            ):

                if not np.any(delta):
                    continue

                score_values = cls._score_fun(X_new.iloc[delta],X_old.iloc[delta])

                # split by direction
                d_dict = {
                    direction: score_values[var_names_dict[direction]].values
                    for direction in var_names_dict.keys()
                }


                # Process the values for each key in the dictionary
                for key in d_dict:
                    if dropped:
                        d_dict[key] = -d_dict[key][~np.isnan(d_dict[key])]
                    else:
                        d_dict[key] = d_dict[key][~np.isnan(d_dict[key])]

                if invert_neg:
                    # Negate the negative values as required
                    d_dict["neg"] = -d_dict["neg"]

                results[concept_name].update({k: v.mean() for k, v in d_dict.items()})

            results = {k:v for k,v in results.items() if len(v) > 0}

            return results
