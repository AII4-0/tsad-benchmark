from torch import Tensor


def adjust_pred(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculate adjusted predicted labels using the ground truth and the prediction.

    :param y_true: The ground truth.
    :param y_pred: The prediction.
    :return: The adjusted predicted labels.
    """
    y_pred_adj = y_pred.clone()
    anomaly_state = False
    for i in range(len(y_pred_adj)):
        if y_true[i] and y_pred_adj[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not y_true[j]:
                    break
                else:
                    if not y_pred_adj[j]:
                        y_pred_adj[j] = True
        elif not y_true[i]:
            anomaly_state = False
        if anomaly_state:
            y_pred_adj[i] = True
    return y_pred_adj
