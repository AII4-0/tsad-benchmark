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
                 train: bool,
                 start_index_inputs_exported_in_c: int = 0,
                 n_inputs_exported_in_c: int = 0) -> None:
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
        self._start_index_inputs_exported_in_c = start_index_inputs_exported_in_c
        self._n_inputs_exported_in_c = n_inputs_exported_in_c

        # Export the data in C
        if self._n_inputs_exported_in_c > 0:
            # Adapat _n_inputs_exported_in_c if the number of input is higher than the number of inputs of the dataset
            self._n_inputs_exported_in_c = (self._features.size(0) - self._start_index_inputs_exported_in_c) if (self._start_index_inputs_exported_in_c + self._n_inputs_exported_in_c) > self._features.size(0) else self._n_inputs_exported_in_c

            # Loop on the selected scv file
            path_scv_file = os.path.join(data_dir, dataset.name, entity + "." + ("train" if train else "test") + ".csv")

            # create an empty file
            file_name = os.path.split(os.path.splitext(path_scv_file)[0])[-1].replace(".", "_").replace("-", "_")
            path_h_file = os.path.split(path_scv_file)[0]
            path_h_file = os.path.join(path_h_file, file_name + ".h")
            open(path_h_file, 'w').close()

            # Input shape : 1 x WindowSize x 1  or  1 x WindowSize x 38
            # Label :       1 x WindowSize      or  1 x WindowSize
            # append the raw data to the file
            with open(path_h_file, "a") as file:
                file_name = os.path.splitext(os.path.split(path_h_file)[-1])[0]
                file.write("\n#ifndef " + file_name + "_H_\n#define " + file_name + "_H_\n\n")

                # Save the dataset values
                file.write("const float " + file_name + "[" + str(self._n_inputs_exported_in_c) + "][" + str(window_size) + "][" + str(dataset.dimension) + "] = {\n")

                # array[iInputs][iWindows][iValues]
                for iInputs in range(self._n_inputs_exported_in_c): # Save the inputs values (iWindows)
                    file.write("{\n")
                    for iWindows in range(window_size):   # Save windows values (iValues)
                        file.write("{")
                        np.savetxt(file, self._features[iInputs + self._start_index_inputs_exported_in_c][iWindows], fmt='%f', delimiter=',', newline=',')
                        file.write("},\n")
                    file.write("},\n")
                file.write("};\n")

                # Save the labels values
                file.write("const float " + file_name + "_label" + "[" + str(self._n_inputs_exported_in_c) + "][" + str(window_size) + "] = {\n")

                # array[iInputs][iWindows]
                for iInputs in range(self._n_inputs_exported_in_c): # Save the inputs values (iWindows)
                    file.write("{\n")
                    np.savetxt(file, self._labels[iInputs + self._start_index_inputs_exported_in_c], fmt='%f', delimiter=',', newline=',\n')
                    file.write("},\n")
                file.write("};\n")

                file.write("const unsigned long " + file_name + "_nInputs = " + str(self._n_inputs_exported_in_c) + ";\n")
                file.write("const unsigned long " + file_name + "_window_size = " + str(window_size) + ";\n")
                file.write("const unsigned long " + file_name + "_dimension = " + str(dataset.dimension) + ";\n")
                file.write("\n#endif\n")
            print("Done: " + file_name)


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
