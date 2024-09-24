from conceptlab.evaluation._base import EvaluationClass
import pandas as pd
from typing import Dict, Any, Literal, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, roc_curve, roc_auc_score, cohen_kappa_score



def cohens_d(x, y):
    """
    Calculate Cohen's d for each column between two NxD arrays.
    Args:
    x: A numpy array of shape (N, D) representing the first group.
    y: A numpy array of shape (N, D) representing the second group.

    Returns:
    A numpy array of Cohen's d values for each column.
    """
    # Means of each column for both groups
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)

    # Pooled standard deviation
    pooled_std = np.sqrt(((np.std(x, axis=0, ddof=1) ** 2) + (np.std(y, axis=0, ddof=1) ** 2)) / 2)

    # Cohen's d calculation for each column
    d_values = (mean_x - mean_y) / pooled_std

    return d_values


def _concept_score(
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

def concept_score(pvals: np.ndarray, direction: Literal["pos", "neg", "neu"], alpha: float = 0.05):

    fun = np.vectorize(lambda x: _concept_score(x, direction, alpha))

    return fun(pvals)




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

        # set axis to compute statistic along
        axis = 0

        if dropped:
            return wilcoxon(Y, X, alternative=direction_alternatives[direction],axis=axis)

        return wilcoxon(X, Y, alternative=direction_alternatives[direction],axis=axis)

    @classmethod
    def score(
        cls,
        X_old: pd.DataFrame,
        X_new: pd.DataFrame,
        concepts_old: pd.DataFrame,
        concepts_new: pd.DataFrame,
        concept_coefs: pd.DataFrame,
            use_neutral: bool = False,
    ) -> Tuple[Dict[str, Any],Dict[str,Any]]:
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

            use_neutral: (bool):  Use shared neutral genes (across all concepts)

        Returns:
            results (Dict[str, Any]): A dictionary containing the scores.
            curves (Dict[str, Any]): A dictionary with the auroc and auprc curves
        """

        concept_names = concepts_old.columns
        concept_uni_vars = (concept_coefs != 0).sum(axis=0)
        concept_neutral_vars = concept_coefs.columns[concept_uni_vars.values == 0]

        direction_to_true = dict(neu = 0,
                                 pos = 1,
                                 neg = 1,
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

            results[concept_name] = dict()

            var_names_dict = dict()

            coefs_k = concept_coefs.loc[concept_name, :]

            var_names_dict["pos"] = coefs_k.index[coefs_k.values > 0]
            var_names_dict["neg"] = coefs_k.index[coefs_k.values < 0]

            if use_neutral:
                var_names_dict["neu"] = concept_neutral_vars
            else:
                var_names_dict["neu"] = coefs_k.index[coefs_k.values == 0]


            pred_vals  = dict()
            true_vals = dict()

            for direction, var_names in var_names_dict.items():
                pred_vals[direction] = []
                true_vals[direction] = []

                vals_new = X_new.loc[:, var_names].values
                vals_old = X_old.loc[:, var_names].values

                pos_delta = concept_delta > 0
                neg_delta = concept_delta < 0

                for delta, dropped, ivn in zip([pos_delta,neg_delta],[False,True],['0->1','1->0']):

                    if np.any(delta):
                        test_res = cls.diff_test(
                            vals_new[delta], vals_old[delta], direction, dropped = dropped,
                        )


                        pred_vals[direction] += test_res.tolist()
                        base = [direction_to_true[direction]] * len(test_res)
                        true_vals[direction] += base


                        diff_av = cohens_d(vals_new[delta], vals_old[delta]).mean()

                        results[concept_name][f"{ivn} : {direction} cohen's d"] =  diff_av

            true_vals_all = np.concatenate([v for v in true_vals.values()])
            pred_vals_all = np.concatenate([v for v in pred_vals.values()])

            # precision recall calculations
            precision, recall, thresholds = precision_recall_curve(true_vals_all, pred_vals_all)
            results[concept_name]['auprc_joint'] =  auc(recall,precision)
            curves[concept_name]['auprc_joint'] = dict(y = precision, x = recall)

            # roc calculations
            fpr, tpr,_ = roc_curve(true_vals_all, pred_vals_all)
            results[concept_name]['auroc_joint'] =  roc_auc_score(true_vals_all, pred_vals_all)
            curves[concept_name]['auroc_joint'] = dict(y = tpr, x = fpr)


        results = {key:val for key,val in results.items() if len(val) > 0}
        curves = {key:val for key,val in curves.items() if len(val) > 0}


        return results,curves

    @classmethod
    def diff_test(cls, X, Y, direction, dropped=False):

        delta = np.mean(X - Y,axis =0)

        if dropped:
            delta = - delta
        if direction == 'neg':
            delta = -delta

        return delta

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
