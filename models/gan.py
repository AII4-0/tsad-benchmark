from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor
from torch.optim import Optimizer

from lightning.entity_module import EntityModule


class Generator(nn.Module):
    """This class is a Generator model for the GAN."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 n: int,
                 n_hidden: int) -> None:
        """
        Create an object of `Generator` class.

        :param n: The size of the 1st layer
        :param n_hidden: The size of the hidden state.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        gen_out = self.model(x)
        return gen_out


class Discriminator(nn.Module):
    """This class is a Discriminator model for the GAN."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 n: int,
                 n_hidden: int) -> None:
        """
        Create an object of `Discriminator` class.

        :param in_channels: the number of channels of the time series
        :param n: The size of the 1st layer
        :param n_hidden: The size of the hidden state.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, in_channels), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        disc_out = self.model(x)
        return disc_out


class GAN(EntityModule):
    """This class is a GAN-based anomaly detection model."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 window_size: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float,
                 **kwargs) -> None:
        """
        Create an object of `GAN` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden state.
        :param window_size: The size of the windows used to predict.
        :param dropout: The probability of an element to be zeroed in the dropout layer.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.n_hidden = hidden_size
        self.n = in_channels * (window_size - 1)

        # Create the generator
        self.generator = Generator(self.n, self.n_hidden)

        # Create the discriminator
        self.discriminator = Discriminator(in_channels, self.n, self.n_hidden)

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
        parser = parent_parser.add_argument_group("GAN")
        parser.add_argument("--hidden_size", type=int, required=True)
        parser.add_argument("--dropout", type=float, required=False)
        parser.add_argument("--num_layers", type=int, required=False)
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        # Apply the GAN
        generator_out = self.generator(x)
        fake_score = self.discriminator(generator_out)

        # Convert the shape to (batch_size, prediction_length, in_channels)
        return fake_score.view(x.size(0), self._prediction_length, -1)

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
