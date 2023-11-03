import torch
from torch import Tensor

from metrics.metrics import f1_score
from metrics.point_adjustment import adjust_pred


def best_threshold(y_true: Tensor, y_pred: Tensor, point_adjustment: bool = False) -> Tensor:
    """
    Search for the best threshold.

    :param y_true: The ground truth.
    :param y_pred: The prediction.
    :param point_adjustment: If the point adjustment should be applied.
    :return: The value of the best threshold.
    """
    # Define variables to store the best threshold and the best score
    best_th = torch.tensor(0.0)
    best_score = torch.tensor(0.0)

    # Define the search interval
    search_interval = torch.linspace(0, 1, 100, device=y_pred.device)

    # Iterate over the search interval [0, 1)
    for anomaly_percent in search_interval:
        # Compute the threshold using `torch.quantile`
        threshold = torch.quantile(y_pred, 1 - anomaly_percent)

        # Binarize the predictions with the threshold
        pred = torch.where(y_pred >= threshold, 1, 0)

        # Perform the point adjustment if needed
        if point_adjustment:
            pred = adjust_pred(y_true, pred)

        # Compute the metric score
        score = f1_score(y_true, pred)

        # If the score is better, update the best threshold and the best score
        if score > best_score:
            best_th = threshold
            best_score = score

    return best_th
