from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F # noqa

from torch import Tensor
from torch.optim import Optimizer

import torch.distributions

from lightning.entity_module import EntityModule


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(nn.Module):
    """This class is a Variational Encoder model for the VAE."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 encoded_size: int) -> None:
        """
        Create an object of `VariationalEncoder` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden layers.
        """
        super().__init__()

        # Create the Linear models
        self.linear1 = nn.Linear(in_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, encoded_size)
        self.linear3 = nn.Linear(hidden_size, encoded_size)

        # Create the distribution
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    """This class is a Decoder model for the VAE."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 encoded_size: int) -> None:
        """
        Create an object of `VariationalEncoder` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden layers.
        """
        super().__init__()

        # Create the Linear models
        self.linear1 = nn.Linear(encoded_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, in_channels)

    def forward(self, z: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param z: The encoded input tensor.
        :return: The output of the model.
        """
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VAE(EntityModule):
    """This class is a VAE-based anomaly detection model."""

    # noinspection PyUnusedLocal
    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 hidden_size: int,
                 encoded_size: int,
                 dropout: float,
                 prediction_length: int,
                 learning_rate: float,
                 **kwargs) -> None:
        """
        Create an object of `VAE` class.

        :param in_channels: The number of channels of the time series.
        :param window_size: The size of the window.
        :param hidden_size: The size of the hidden layers.
        :param dropout: The probability of an element to be zeroed in the dropout layer.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Create the Encoder
        self.encoder = VariationalEncoder(in_channels * (window_size - 1), hidden_size, encoded_size)

        # Create the Decoder
        self.decoder = Decoder(in_channels, hidden_size)

        # Store `prediction_length` as private class attribute
        self._prediction_length = prediction_length

        # Store `learning_rate` as private class attribute
        self._learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        """
        Implement the forward pass of the model.

        :param x: The input tensor.
        :return: The output of the model.
        """
        # Apply the VAE
        x = x.to(device)
        z = self.decoder(self.encoder(x))

        # Convert the shape to (batch_size, prediction_length, in_channels)
        return z.view(x.size(0), self._prediction_length, -1)

    def add_argparse_args(self, parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("VAE")
        parser.add_argument("--hidden_size", type=int, required=True)
        parser.add_argument("--encoded_size", type=int, required=True)
        parser.add_argument("--dropout", type=float, required=True)
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)

        return parent_parser

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
