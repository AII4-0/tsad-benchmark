import os
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch.utils.data import Dataset

from datasets.dataset_descriptor import DatasetDescriptor


class TimeSeriesDataset(Dataset):
    """This class is an implementation of the PyTorch's `Dataset` for time series data."""

    def __init__(self,
                 data_dir: str,
                 dataset: DatasetDescriptor,
                 entity: str,
                 scaler: Union[MinMaxScaler, StandardScaler, RobustScaler],
                 window_size: int,
                 train: bool) -> None:
        """
        Create an object of the `TimeSeriesDataset` class.

        :param data_dir: The path to where the data is saved.
        :param dataset: The dataset descriptor of the dataset that should be used.
        :param entity: The entity of the dataset that should be used.
        :param scaler: The scaler that should be used.
        :param window_size: The size of the sliding window.
        :param train: A flag to know if it is the train or the test set.
        """
        # Build the path to the CSV file
        path = os.path.join(data_dir, dataset.name, entity + "." + ("train" if train else "test") + ".csv")

        # Load data from CSV file
        data = pd.read_csv(path)

        # Convert float64 columns to float32
        data = data.astype({c: np.float32 for c in data.select_dtypes(include="float64").columns})

        # Drop timestamp column
        data = data.drop(columns="timestamp")

        # Separate the features and the labels
        labels = data.is_anomaly
        features = data.drop(columns="is_anomaly")

        # Normalize the features
        if train:
            features[features.columns] = scaler.fit_transform(features[features.columns])
        else:
            features[features.columns] = scaler.transform(features[features.columns])

        # Generate windows
        features = np.lib.stride_tricks.sliding_window_view(features.values, window_size, axis=0).transpose((0, 2, 1))
        labels = np.lib.stride_tricks.sliding_window_view(labels.values, window_size, axis=0)

        # Set as class attributes
        self._features = torch.tensor(features)
        self._labels = torch.tensor(labels)
        self._train = train

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return the specified item of the dataset.

        :param index: The index of the dataset item.
        :return: The specified item.
        """
        if self._train:
            return self._features[index]
        else:
            return self._features[index], self._labels[index]

    def __len__(self) -> int:
        """
        Return the number of items contained in the dataset.

        :return: The number of items.
        """
        return self._features.size(0)
