from argparse import ArgumentParser
from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor

from benchmark.model_module import ModelModule


class Transformer(ModelModule):
    """This class is a Transformer-based anomaly detection model."""

    def __init__(self,
                 in_channels: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float) -> None:
        """
        Create an object of `Transformer` class.

        :param in_channels: The number of channels of the time series.
        :param num_encoder_layers : The number of layers of the encoder model.
        :param num_decoder_layers : The number of layers of the decoder model.
        :param dropout: The probability of an element to be zeroed in the dropout layer.
        """
        super().__init__(prediction_length, learning_rate)

        # Create the Transformer
        self.transformer = nn.Transformer(
            d_model=in_channels,
            nhead=1,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        # Create the dropout
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("Transformer")
        parser.add_argument("--num_encoder_layers", type=int, required=True)
        parser.add_argument("--num_decoder_layers", type=int, required=True)
        parser.add_argument("--dropout", type=float, required=True)
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        return parent_parser

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :param y: The label tensor.
        :return: The output of the model.
        """
        # Apply the Transformer
        transformer_out = self.transformer(x, y)

        # Apply the dropout
        dropout_out = self.dropout(transformer_out[:, -1])

        # Convert the shape to (batch_size, prediction_length, in_channels)
        return dropout_out.view(x.size(0), self._prediction_length, -1)

    def training_step(self, batch: Tensor) -> Tensor:
        """
        Perform a training step.

        :param batch: The batch data.
        :return: The loss of the batch.
        """
        return super()._training_step(batch, pass_labels=True)

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Perform a test step.

        :param batch: The batch data (features, labels).
        :return: The test step output (predictions, labels).
        """
        return super()._test_step(batch, pass_labels=True)
