from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor
from torch.optim import Optimizer

from lightning.entity_module import EntityModule


class LSTM(EntityModule):
    """This class is a LSTM-based anomaly detection model."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float,
                 **kwargs) -> None:
        """
        Create an object of `LSTM` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden state.
        :param num_layers: The number of layers of the model.
        :param dropout: The probability of an element to be zeroed in the dropout layer.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Create the LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Create the dropout
        self.dropout = nn.Dropout(dropout)

        # Create the predictor
        self.predictor = nn.Linear(hidden_size, prediction_length * in_channels)

        # Store `prediction_length` as private class attribute
        self._prediction_length = prediction_length

        # Store `learning_rate` as private class attribute
        self._learning_rate = learning_rate

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("LSTM")
        parser.add_argument("--hidden_size", type=int, required=True)
        parser.add_argument("--num_layers", type=int, required=True)
        parser.add_argument("--dropout", type=float, required=True)
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        # Apply the LSTM
        lstm_out, _ = self.lstm(x)

        # Apply the dropout
        dropout_out = self.dropout(lstm_out[:, -1])

        # Apply the predictor
        pred_out = self.predictor(dropout_out)

        # Convert the shape to (batch_size, prediction_length, in_channels)
        return pred_out.view(x.size(0), self._prediction_length, -1)

    def training_step(self, batch: Tensor) -> Tensor:
        """
        Perform a training step.

        :param batch: The batch data.
        :return: The loss of the batch.
        """
        # Split x and y
        x = batch[:, : -self._prediction_length]
        y = batch[:, -self._prediction_length:]

        # Apply the models
        y_hat = self(x)

        # Compute the loss
        loss = F.mse_loss(y_hat, y, reduction="sum")

        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Perform a test step.

        :param batch: The batch data (features, labels).
        :param batch_idx: The batch index.
        :return: The test step output (predictions, labels).
        """
        # Unpack the batch
        features, labels = batch

        # Split x and y
        x = features[:, : -self._prediction_length]
        y = features[:, -self._prediction_length:]

        # Apply the models
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
