from typing import List, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from utils.metrics import confusion_matrix, f1_score_from_confusion_matrix, precision_from_confusion_matrix, \
    recall_from_confusion_matrix
from utils.point_adjustment import adjust_pred
from utils.thresholding import best_threshold


class EntityModule(LightningModule):
    """This class implements the `LightningModule` callbacks that are common to all models."""

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """
        Log the last loss value of the training epoch.

        :param outputs: The outputs of the training steps.
        """
        # Log the last loss value
        self.log_dict(outputs[-1])

    def test_epoch_end(self, outputs: List[Tuple[Tensor, Tensor]]) -> None:
        """
        Compute the confusion matrix and the F1 score at end of a test epoch.

        :param outputs: The outputs of the test steps.
        """
        # Convert the list of tuples to a list of two lists
        list_of_lists = list(map(list, zip(*outputs)))

        # Unpack the list and convert to tensors
        y_pred = torch.cat(list_of_lists[0])
        y_true = torch.cat(list_of_lists[1])

        # Find the best thresholds
        threshold = best_threshold(y_true, y_pred)
        threshold_adj = best_threshold(y_true, y_pred, point_adjustment=True)

        # Binarize the predictions with the corresponding threshold
        y_pred_std = torch.where(y_pred >= threshold, 1, 0)
        y_pred_adj = torch.where(y_pred >= threshold_adj, 1, 0)

        # Apply the point adjustment to the prediction
        y_pred_adj = adjust_pred(y_true, y_pred_adj)

        # Compute the confusion matrices
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred_std)
        tp_adj, tn_adj, fp_adj, fn_adj = confusion_matrix(y_true, y_pred_adj)

        # Compute the precision
        precision = precision_from_confusion_matrix(tp, tn, fp, fn)
        precision_adj = precision_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

        # Compute the recall
        recall = recall_from_confusion_matrix(tp, tn, fp, fn)
        recall_adj = recall_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

        # Compute the F1 scores
        f1_score = f1_score_from_confusion_matrix(tp, tn, fp, fn)
        f1_score_adj = f1_score_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

        # Log the values
        self.log("threshold", threshold)
        self.log("threshold_adjusted", threshold_adj)
        self.log("true_positives", tp.float())
        self.log("true_positives_adjusted", tp_adj.float())
        self.log("true_negatives", tn.float())
        self.log("true_negatives_adjusted", tn_adj.float())
        self.log("false_positives", fp.float())
        self.log("false_positives_adjusted", fp_adj.float())
        self.log("false_negatives", fn.float())
        self.log("false_negatives_adjusted", fn_adj.float())
        self.log("precision", precision)
        self.log("precision_adjusted", precision_adj)
        self.log("recall", recall)
        self.log("recall_adjusted", recall_adj)
        self.log("f1", f1_score)
        self.log("f1_adjusted", f1_score_adj)
