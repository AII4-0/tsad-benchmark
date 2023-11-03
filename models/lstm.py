from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor

from benchmark.model_module import ModelModule


class LSTM(ModelModule):
    """This class is a LSTM-based anomaly detection model."""

    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float) -> None:
        """
        Create an object of `LSTM` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden state.
        :param num_layers: The number of layers of the model.
        :param dropout: The probability of an element to be zeroed in the dropout layer.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        """
        super().__init__(prediction_length, learning_rate)

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
