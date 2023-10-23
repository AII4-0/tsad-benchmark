from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor
from torch.optim import Optimizer

from lightning.entity_module import EntityModule


class TranAD(EntityModule):
    """This class is a TranAD-based anomaly detection model."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 window_size: int,
                 num_layers: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float,
                 **kwargs) -> None:
        """
        Create an object of `TranAD` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden state.
        :param window_size: The size of the window used to predict.
        :param num_layers: The number of layers for the model
        :parma dropout: The dropout rate of the model
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * in_channels,
            nhead=1,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        # Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )

        # Decoder layer 1
        self.decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=2 * in_channels,
            nhead=1,
            dim_feedforward=hidden_size,
            activation="relu",
            batch_first=True
        )

        # Decoder 1
        self.transformer_decoder1 = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer1,
            num_layers=num_layers
        )

        # Decoder layer 2
        self.decoder_layer2 = nn.TransformerDecoderLayer(
            d_model=2 * in_channels,
            nhead=1,
            dim_feedforward=hidden_size,
            activation="relu",
            batch_first=True
        )

        # Decoder 2
        self.transformer_decoder2 = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer2,
            num_layers=num_layers
        )

        # FCN
        self.fcn = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.Sigmoid()
        )

        # Store `prediction_length` as private class attribute
        self._prediction_length = prediction_length

        # Store `learning_rate` as private class attribute
        self._learning_rate = learning_rate

        # Store `_in_channels` as private class attribute
        self._in_channels = in_channels

    def encode(self, src: Tensor, c: Tensor, tgt: Tensor) -> tuple[Tensor, any]:
        """
        Encode the source Tensor.

        :param src: The source (parameters) Tensor.
        :param c: The anomaly score
        :param tgt: The target (label) Tensor
        :return: The encoded Tensor and the target.
        """
        src = torch.cat((src, c), dim=2)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("TranAD")
        parser.add_argument("--hidden_size", type=int, required=True)
        parser.add_argument("--num_layers", type=int, required=False)
        parser.add_argument("--dropout", type=float, required=False)
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        return parent_parser

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param src: The input tensor.
        :param tgt: The output tensor.
        :return: The output of the model.
        """
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

        # Convert the shape to (batch_size, prediction_length, in_channels)
        return x2.view(src.size(0), self._prediction_length, -1)

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
        y_hat = self(x, y)

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
        y_hat = self(x, y)

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
