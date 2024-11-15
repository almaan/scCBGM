from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
)
import pandas as pd
import numpy as np


def concept_accuarcy(concepts, pred_concept, debug=False):

    n_concept = concepts.shape[1]
    loss_dict = {}
    all_acc = 0
    all_f1 = 0
    all_auprc = 0
    all_auroc = 0

    if isinstance(concepts, pd.DataFrame):
        concepts = concepts.values

    threshold = 0.5
    n_valid_concepts = 0

    for c in range(n_concept):
        c_true = concepts[:, c]
        c_pred = pred_concept[:, c]

        n_true = np.sum(c_true)
        n_false = c_true.shape[0] - n_true

        c_pred_bin = (c_pred >= threshold).astype(int)

        if (n_true > 0) and (n_false > 0):

            n_valid_concepts += 1

            # Calculate F1 score
            f1 = f1_score(c_true, c_pred_bin)
            all_f1 += f1

            # Calculate AUPRC
            precision, recall, _ = precision_recall_curve(c_true, c_pred)
            auprc = auc(recall, precision)
            all_auprc += auprc

            # Calculate AUROC directly
            auroc = roc_auc_score(c_true, c_pred)
            all_auroc += auroc
        else:
            auroc, auprc, f1 = None, None, None

        correct_predictions = (c_pred_bin == c_true).sum().item()
        total_predictions = c_true.shape[0]
        accuracy = correct_predictions / total_predictions
        all_acc += accuracy

    if debug:
        loss_dict["test_concept_" + str(c) + "_acc"] = accuracy

        if f1 is not None:
            loss_dict["test_concept_" + str(c) + "_f1"] = f1
        if auprc is not None:
            loss_dict["test_concept_" + str(c) + "_auprc"] = auprc
        if auroc is not None:
            loss_dict["test_concept_" + str(c) + "_auroc"] = auroc

    loss_dict["test_avg_concept_acc"] = all_acc / n_concept
    loss_dict["test_avg_concept_f1"] = all_f1 / max(n_valid_concepts, 1)
    loss_dict["test_avg_concept_auprc"] = all_auprc / max(n_valid_concepts, 1)
    loss_dict["test_avg_concept_auroc"] = all_auroc / max(n_valid_concepts, 1)

    return loss_dict
