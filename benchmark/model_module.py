from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor
from torch.optim import Optimizer


class ModelModule(nn.Module):

    def __init__(self,
                 prediction_length: int,
                 learning_rate: float) -> None:
        """
        Create an object of `ModelModule` class.

        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        """
        super().__init__()
        self._prediction_length = prediction_length
        self._learning_rate = learning_rate

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        return parent_parser

    def training_step(self, batch: Tensor) -> Tensor:
        """
        Perform a training step.

        :param batch: The batch data.
        :return: The loss of the batch.
        """
        return self._training_step(batch)

    def _training_step(self, batch: Tensor, pass_labels=False) -> Tensor:
        """
        Perform a training step.

        :param batch: The batch data.
        :param pass_labels: The labels must be passed to the model (default: False).
        :return: The loss of the batch.
        """
        # Split x and y
        x = batch[:, : -self._prediction_length]
        y = batch[:, -self._prediction_length:]

        # Apply the models
        if pass_labels:
            y_hat = self(x, y)
        else:
            y_hat = self(x)

        # Compute the loss
        loss = F.mse_loss(y_hat, y, reduction="sum")

        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Perform a test step.

        :param batch: The batch data (features, labels).
        :return: The test step output (predictions, labels).
        """
        return self._test_step(batch)

    def _test_step(self, batch: Tuple[Tensor, Tensor], pass_labels=False) -> Tuple[Tensor, Tensor]:
        """
        Perform a test step.

        :param batch: The batch data (features, labels).
        :param pass_labels: The labels must be passed to the model (default: False).
        :return: The test step output (predictions, labels).
        """
        # Unpack the batch
        features, labels = batch

        # Split x and y
        x = features[:, : -self._prediction_length]
        y = features[:, -self._prediction_length:]

        # Apply the models
        if pass_labels:
            y_hat = self(x, y)
        else:
            y_hat = self(x)

        # Compute the score
        score = F.mse_loss(y_hat, y, reduction="none")

        # Averaging of all variates (batch_size, prediction_length, variates) -> (batch_size, prediction_length)
        score = score.mean(dim=-1)

        # Apply a sigmoid
        score = score.sigmoid()

        return score.mean(dim=-1), torch.where(labels.sum(dim=-1) > 0, 1, 0)

    def configure_optimizers(self) -> Optimizer:
        """
        Define the optimizer for the training.

        :return: The optimizer for the training.
        """
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate, weight_decay=0.001)
