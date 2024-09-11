from conceptlab.evaluation._base import EvaluationClass
import pandas as pd
from typing import Dict, Any, Literal
import numpy as np


def concept_score(
    pval: float, direction: Literal["pos", "neg", "neu"], alpha: float = 0.05
):
    """
        Compute a score based on the p-value and the specified direction.

        Parameters:
        -----------
        pval : float
            The p-value to be compared against the significance threshold `alpha`.

        direction : Literal["pos", "neg", "neu"]
            The direction of the test. Can be:
                - "pos" or "neg": If the p-value is less than `alpha`, the score is 1.0; otherwise, 0.0.
                - "neu": If the p-value is greater than `alpha`, the score is 1.0; otherwise, 0.0.

        alpha : float, optional, default=0.05
            The significance threshold. This is the value against which the p-value is compared.

        Returns:
        --------
        float
            A score of 1.0 or 0.0 depending on the comparison between the p-value and `alpha`.
            - For "pos" or "neg" directions, 1.0 if `pval < alpha`, else 0.0.
            - For "neu", 1.0 if `pval > alpha`, else 0.0.

        Raises:
        -------
        ValueError
            If the `direction` is not one of "pos", "neg", or "neu".
    """

    match direction:
        case "pos" | "neg":
            score = float(pval < alpha)
        case "neu":
            score = float(pval > alpha)
        case _:
            raise ValueError("Incorrect Direction")
    return score



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
        use_neutral: bool = True,
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
            use_neutral: (bool): Only use genes for neutral that are not impacted by _any_ concept

        Returns:
            results (Dict[str, Any]): A dictionary containing the statistical
                                      test results for each concept shift.
        """

        concept_names = concepts_old.columns
        concept_uni_vars = (concept_coefs != 0).sum(axis=0)

        concept_neutral_vars = concept_coefs.columns[concept_uni_vars.values == 0]

        results = dict()
        scores = dict()

        for concept_name in concept_names:

            concept_delta = (
                concepts_new.loc[:, concept_name] - concepts_old.loc[:, concept_name]
            ).values

            concept_delta_mean = concept_delta.mean()

            if concept_delta_mean == 0:
                continue

            results[concept_name] = dict()
            scores[concept_name] = dict()

            var_names_dict = dict()

            coefs_k = concept_coefs.loc[concept_name, :]

            var_names_dict["pos"] = coefs_k.index[coefs_k.values > 0]
            var_names_dict["neg"] = coefs_k.index[coefs_k.values < 0]
            if use_neutral:
                var_names_dict["neu"] = concept_neutral_vars
            else:
                var_names_dict["neu"] = coefs_k.index[coefs_k.values == 0]

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
                    scores[concept_name][f"0->1 : {direction}"] = concept_score(
                        stat_res[1], direction
                    )

                neg_delta = concept_delta < 0
                if np.any(neg_delta):
                    stat_res = cls.stat_test(
                        vals_new[neg_delta],
                        vals_old[neg_delta],
                        direction,
                        dropped=True,
                    )

                    results[concept_name][f"1->0 : {direction}"] = stat_res[1]
                    scores[concept_name][f"1->0 : {direction}"] = concept_score(
                        stat_res[1], direction
                    )

        return results, scores

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
