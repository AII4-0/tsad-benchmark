from argparse import ArgumentParser

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor

from benchmark.model_module import ModelModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(nn.Module):
    """This class is a Variational Encoder model for the VAE."""

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


class VAE(ModelModule):
    """This class is a VAE-based anomaly detection model."""

    def __init__(self,
                 in_channels: int,
                 window_size: int,
                 hidden_size: int,
                 encoded_size: int,
                 prediction_length: int,
                 learning_rate: float) -> None:
        """
        Create an object of `VAE` class.

        :param in_channels: The number of channels of the time series.
        :param window_size: The size of the window.
        :param hidden_size: The size of the hidden layers.
        :param encoded_size: The size of encoded tensors.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        """
        super().__init__(prediction_length, learning_rate)

        # Create the Encoder
        self.encoder = VariationalEncoder(in_channels * (window_size - 1), hidden_size, encoded_size)

        # Create the Decoder
        self.decoder = Decoder(in_channels, hidden_size, encoded_size)

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
        parser.add_argument("--prediction_length", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        return parent_parser
