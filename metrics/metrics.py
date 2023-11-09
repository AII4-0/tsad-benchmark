from typing import Tuple

import torch
from torch import Tensor

# Define the epsilon
EPS = torch.finfo(torch.float32).eps


def confusion_matrix(y_true: Tensor, y_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the confusion matrix.

    :param y_true: The ground truth.
    :param y_pred: The prediction.
    :return: A tuple containing the confusion matrix components (TP, TN, FP, FN).
    """
    true_positives = torch.sum(y_pred * y_true)
    true_negatives = torch.sum((1 - y_pred) * (1 - y_true))
    false_positives = torch.sum(y_pred * (1 - y_true))
    false_negatives = torch.sum((1 - y_pred) * y_true)
    return true_positives, true_negatives, false_positives, false_negatives


def f1_score_from_confusion_matrix(tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
    """
    Compute the F1 score using the confusion matrix components.

    :param tp: The number of true positives.
    :param tn: The number of true negatives.
    :param fp: The number of false positives.
    :param fn: The number of false negatives.
    :return: The F1 score.
    """
    return (2 * tp) / (2 * tp + fp + fn + EPS)


def precision_from_confusion_matrix(tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
    """
    Compute the precision using the confusion matrix components.

    :param tp: The number of true positives.
    :param tn: The number of true negatives.
    :param fp: The number of false positives.
    :param fn: The number of false negatives.
    :return: The precision.
    """
    return tp / (tp + fp + EPS)


def recall_from_confusion_matrix(tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
    """
    Compute the recall using the confusion matrix components.

    :param tp: The number of true positives.
    :param tn: The number of true negatives.
    :param fp: The number of false positives.
    :param fn: The number of false negatives.
    :return: The recall.
    """
    return tp / (tp + fn + EPS)


def f1_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Compute the F1 score using the ground truth and the prediction.

    :param y_true: The ground truth.
    :param y_pred: The prediction.
    :return: The F1 score.
    """
    return f1_score_from_confusion_matrix(*confusion_matrix(y_true, y_pred))
