from conceptlab.evaluation._base import EvaluationClass
import pandas as pd
from typing import Dict, Any
import numpy as np


def concept_score(data, pvalue=0.05,comparison_list=[1,1,0]):
    """
    Compare dictionary values to a pvalue and return a list of 1s and 0s.
    
    Parameters:
    - data (dict): Dictionary with numerical values.
    - pvalue (float): Threshold for comparison.
    
    Returns:
    - list: List of 1s and 0s based on the comparison.
    """
    threshold = [1 if value < pvalue else 0 for value in data.values()]
    return threshold

class DistributionShift(EvaluationClass):
    """
    A class to evaluate distribution shifts between two datasets by comparing 
    their statistical properties and concept distributions.

    Inherits from:
        EvaluationClass: A base class that provides common evaluation utilities.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def stat_test(cls, X, Y, direction, dropped=False):
        """
        Performs a Wilcoxon signed-rank test between two samples, X and Y,
        to evaluate if there is a significant difference between them
        based on the specified direction.

        Args:
            X (array-like): The first sample.
            Y (array-like): The second sample.
            direction (str): The direction of the test, one of 'pos' (greater),
                             'neg' (less), or 'neu' (two-sided).
            dropped (bool, optional): Whether the intervention was to drop or add the concept
                                      Default is False.

        Returns:
            stat (float): The test statistic.
            p-value (float): The p-value of the test.

        """
        from scipy.stats import wilcoxon

        direction_alternatives = dict(
            pos="greater",
            neg="less",
            neu="two-sided",
        )

        if dropped:
            return wilcoxon(Y, X, alternative=direction_alternatives[direction])

        return wilcoxon(X, Y, alternative=direction_alternatives[direction])

    @classmethod
    def score(
        cls,
        X_old: pd.DataFrame,
        X_new: pd.DataFrame,
        concepts_old: pd.DataFrame,
        concepts_new: pd.DataFrame,
        concept_coefs: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluates the distribution shift by calculating statistical 
        significance of changes in concept distributions between 
        the old and new datasets.

        Args:
            X_old (pd.DataFrame): The original dataset features.
            X_new (pd.DataFrame): The new dataset features.
            concepts_old (pd.DataFrame): Concept values in the original dataset.
            concepts_new (pd.DataFrame): Concept values in the new dataset.
            concept_coefs (pd.DataFrame): Coefficients indicating the relationship 
                                          between variables and concepts.

        Returns:
            results (Dict[str, Any]): A dictionary containing the statistical 
                                      test results for each concept shift.
        """

        concept_names = concepts_old.columns
        concept_uni_vars = (concept_coefs != 0).sum(axis=0)
        concept_neutral_vars = concept_coefs.columns[concept_uni_vars.values == 0]

        results = dict()

        for concept_name in concept_names:

            concept_delta = (
                concepts_new.loc[:, concept_name] - concepts_old.loc[:, concept_name]
            ).values



            concept_delta_mean = concept_delta.mean()

            if concept_delta_mean == 0:
                continue

            results[concept_name] = dict()

            var_names_dict = dict()

            coefs_k = concept_coefs.loc[concept_name, :]

            var_names_dict["pos"] = coefs_k.index[coefs_k.values > 0]
            var_names_dict["neg"] = coefs_k.index[coefs_k.values < 0]
            var_names_dict["neu"] = concept_neutral_vars

            for direction, var_names in var_names_dict.items():

                if len(var_names) < 1:
                    continue

                vals_new = X_new.loc[:, var_names].mean(axis=1).values
                vals_old = X_old.loc[:, var_names].mean(axis=1).values

                pos_delta = concept_delta > 0

                if np.any(pos_delta):
                    stat_res = cls.stat_test(
                        vals_new[pos_delta], vals_old[pos_delta], direction
                    )
                    results[concept_name][f"0->1 : {direction}"] = stat_res[1]

                neg_delta = concept_delta < 0
                if np.any(neg_delta):
                    stat_res = cls.stat_test(
                        vals_new[neg_delta],
                        vals_old[neg_delta],
                        direction,
                        dropped=True,
                    )
                    results[concept_name][f"1->0 : {direction}"] = stat_res[1]


            results[concept_name]["score"] =  1-np.bitwise_xor(concept_score(results[concept_name]), [1,1,0])
           
        return results

    @classmethod
    def pretty_display(cls, results):
        """
        Displays the results of the distribution shift evaluation in a 
        well-formatted table.

        Args:
            results (Dict[str, Any]): The results dictionary to display.

        Returns:
            None
        """

        with pd.option_context(
            "display.float_format", "{:.2e}".format
        ) and pd.option_context("display.width", 1000):
            print(pd.DataFrame(results).T)
