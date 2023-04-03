# autogluon_custom_metric_serializable.py
import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer

def probabilistic_f1_score(y_true, y_pred_proba, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred_proba, pd.Series):
        y_pred_proba = y_pred_proba.to_numpy()
    for idx in range(len(y_true)):
        prediction = min(max(y_pred_proba[idx], 0), 1)
        if (y_true[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = 0 if y_true_count==0 else ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def probabilistic_f1_score1(y_true, y_pred_proba, beta=1):
    """
    Calculate the probabilistic F1 score based on the paper "Probabilistic Extension of Precision, Recall, and F1 Score".

    :param y_true: true labels
    :param y_pred_proba: predicted probabilities for each class
    :param beta: F1 score beta parameter (default=1)
    :return: probabilistic F1 score
    """
    # Calculate probabilistic precision and recall
    p_tp = np.sum(y_true * y_pred_proba, axis=0)
    p_fn = np.sum(y_true * (1 - y_pred_proba), axis=0)
    p_fp = np.sum((1 - y_true) * y_pred_proba, axis=0)
    p_precision = p_tp / (p_tp + p_fp)
    p_recall = p_tp / (p_tp + p_fn)

    # Calculate probabilistic F1 score
    if p_precision <= 0 or p_recall <= 0:
        p_f1_score = 0
    else:
        p_f1_score = (1 + beta**2) * (p_precision * p_recall) / (beta**2 * p_precision + p_recall)

    return p_f1_score

# Define your custom probabilistic F1 scorer with a specific beta value
beta_value = 1
probabilistic_f1_scorer = make_scorer(name='my_prob_scorer', score_func=probabilistic_f1_score,
                                      optimum=1, greater_is_better=True,
                                      needs_proba=True)


