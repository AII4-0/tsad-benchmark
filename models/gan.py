from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch import Tensor

from benchmark.model_module import ModelModule


class Generator(nn.Module):
    """This class is a Generator model for the GAN."""

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


class GAN(ModelModule):
    """This class is a GAN-based anomaly detection model."""

    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 window_size: int,
                 prediction_length: int,
                 learning_rate: float) -> None:
        """
        Create an object of `GAN` class.

        :param in_channels: The number of channels of the time series.
        :param hidden_size: The size of the hidden state.
        :param window_size: The size of the windows used to predict.
        :param prediction_length: The length of the prediction.
        :param learning_rate: The learning rate.
        """
        super().__init__(prediction_length, learning_rate)

        self.n_hidden = hidden_size
        self.n = in_channels * (window_size - 1)

        # Create the generator
        self.generator = Generator(self.n, self.n_hidden)

        # Create the discriminator
        self.discriminator = Discriminator(in_channels, self.n, self.n_hidden)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("GAN")
        parser.add_argument("--hidden_size", type=int, required=True)
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
