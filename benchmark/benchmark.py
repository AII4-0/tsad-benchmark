from argparse import ArgumentParser
from copy import deepcopy

import torch
from torch import Tensor

from benchmark.data_module import DataModule
from benchmark.model_module import ModelModule
from metrics.metrics import confusion_matrix, f1_score_from_confusion_matrix, precision_from_confusion_matrix, \
    recall_from_confusion_matrix
from metrics.point_adjustment import adjust_pred
from metrics.thresholding import best_threshold


class Benchmark:
    """This class manages the benchmarking of a model."""

    def __init__(self, epochs: int) -> None:
        """
        Create an object of `Trainer` class.

        :param epochs: The number of epochs.
        """
        self._epochs = epochs

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("Benchmark")
        parser.add_argument("--epochs", type=int, required=True)
        return parent_parser

    def run(self, model: ModelModule, data: DataModule) -> None:
        """
        Run a benchmark of a model.

        :param model: The model to benchmark.
        :param data: The data used to benchmark the model.
        """
        # Prepare data
        data.prepare_data()

        # Save the initial weights
        weights = deepcopy(model.state_dict())

        # Initialize global confusion matrix variables
        gl_tp = torch.tensor(0)
        gl_tp_adj = torch.tensor(0)
        gl_tn = torch.tensor(0)
        gl_tn_adj = torch.tensor(0)
        gl_fp = torch.tensor(0)
        gl_fp_adj = torch.tensor(0)
        gl_fn = torch.tensor(0)
        gl_fn_adj = torch.tensor(0)

        # Iterate over entities
        for entity, (train_dataloader, test_dataloader) in enumerate(data):
            # Restore the initial weights
            model.load_state_dict(weights)

            # Get the optimizer
            optimizer = model.configure_optimizers()

            # Iterate over epochs
            for epoch in range(self._epochs):

                # Create a list to store the epoch's losses
                losses = []

                # Iterate over train batches
                for batch in train_dataloader:
                    # Perform a train step
                    loss = model.training_step(batch)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Add the current loss to the list
                    losses.append(loss.item())

                # Logging
                print(f"Entity {entity} | Train epoch {epoch} | Loss: {sum(losses) / len(losses)}")

            # Set the model to eval mode
            model.eval()

            # Disable gradient calculation
            with torch.no_grad():

                # Create lists to store batch results
                y_pred_list = []
                y_true_list = []

                # Iterate over test batches
                for batch in test_dataloader:
                    # Perform the prediction
                    pred, true = model.test_step(batch)

                    # Add results to the lists
                    y_pred_list.append(pred)
                    y_true_list.append(true)

                # Convert lists to tensors
                y_pred = torch.cat(y_pred_list)
                y_true = torch.cat(y_true_list)

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

                # Logging
                print(f"Entity {entity} | Test          | Precision: {precision}, Precision adjusted: {precision_adj}")

                # Compute the recall
                recall = recall_from_confusion_matrix(tp, tn, fp, fn)
                recall_adj = recall_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

                print(f"Entity {entity} | Test          | Recall: {recall}, Recall adjusted: {recall_adj}")

                # Compute the F1 scores
                f1_score = f1_score_from_confusion_matrix(tp, tn, fp, fn)
                f1_score_adj = f1_score_from_confusion_matrix(tp_adj, tn_adj, fp_adj, fn_adj)

                print(f"Entity {entity} | Test          | F1: {f1_score}, F1 adjusted: {f1_score_adj}")

                # Update the global confusion matrix
                gl_tp += tp
                gl_tp_adj += tp_adj
                gl_tn += tn
                gl_tn_adj += tn_adj
                gl_fp += fp
                gl_fp_adj += fp_adj
                gl_fn += fn
                gl_fn_adj += fn_adj

                # Set the model to train mode
                model.train()

        # Compute the global precision
        gl_precision = precision_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_precision_adj = precision_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Compute the global recall
        gl_recall = recall_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_recall_adj = recall_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Compute the global F1 scores
        gl_f1_score = f1_score_from_confusion_matrix(gl_tp, gl_tn, gl_fp, gl_fn)
        gl_f1_score_adj = f1_score_from_confusion_matrix(gl_tp_adj, gl_tn_adj, gl_fp_adj, gl_fn_adj)

        # Print the global metrics
        self._print_global_metrics({
            "Precision": gl_precision,
            "Precision adjusted": gl_precision_adj,
            "Recall": gl_recall,
            "Recall adjusted": gl_recall_adj,
            "F1 score": gl_f1_score,
            "F1 score adjusted": gl_f1_score_adj
        })

    @staticmethod
    def _print_global_metrics(metrics: dict[str, Tensor]) -> None:
        # Compute the max lengths of the key and the value
        max_len_key = max([len(k) for k in metrics.keys()])
        max_len_value = max([len(str(v.numpy())) for v in metrics.values()])

        # Compute the width
        width = 3 * 5 + max_len_key + max_len_value

        # Header
        left_pad = (width - 14) // 2
        print("─" * width)
        print((" " * left_pad) + "Global metrics")
        print("─" * width)

        # Rows
        for k, v in metrics.items():
            v = str(v.numpy())
            left_pad = (max_len_key - len(k)) // 2
            middle_pad = max_len_key - len(k) - left_pad + 5 + (max_len_value - len(v)) // 2
            print("     " + (" " * left_pad) + k + (" " * middle_pad) + v)

        # Footer
        print("─" * width)
