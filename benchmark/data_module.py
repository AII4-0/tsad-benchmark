import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from datasets.time_series_dataset import TimeSeriesDataset
from utils.download_progress_bar import DownloadProgressBar


class DataModule:
    """This class manages the data loading for the benchmark."""

    def __init__(self,
                 data_dir: str,
                 dataset: str,
                 window_size: int,
                 batch_size: int) -> None:
        """
        Create an object of `DataModule` class.

        :param data_dir: The path to where the data is saved.
        :param dataset: The name of the dataset that should be used.
        :param window_size: The size of the sliding window.
        :param batch_size: The size of the batch.
        """
        # Public attribute
        self.dataset = getattr(Datasets, dataset.replace("-", "_"))

        # Private attributes
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._window_size = window_size
        self._batch_size = batch_size
        self._entity_idx = 0

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Extend existing argparse.

        :param parent_parser: The parent `ArgumentParser`.
        :return: The extended argparse.
        """
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--data_dir", type=str, default="data")
        parser.add_argument(
            "--dataset",
            choices=[
                "KDD-TSAD",
                "NASA-MSL",
                "NASA-SMAP",
                "SMD",
                "SWAT",
                "WADI"
            ],
            required=True
        )
        parser.add_argument("--window_size", type=int, required=True)
        parser.add_argument("--batch_size", type=int, required=True)
        return parent_parser

    def prepare_data(self) -> None:
        """Download (only if needed) and prepare data."""
        # Create the data directory if not exists
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        # Build the path to the datasets folder
        path = os.path.join(self._data_dir, self.dataset.name)

        # Check if the datasets is available in local
        if not os.path.exists(path):
            # Build the path to the ZIP file
            zip_file = path + ".zip"

            # Download the datasets
            with DownloadProgressBar(self.dataset.name) as progress_bar:
                urllib.request.urlretrieve(
                    self.dataset.download_url,
                    filename=zip_file,
                    reporthook=progress_bar.update_to
                )

            # Build the path to the temp directory
            temp_dir = path + "-TEMP"

            # Unzip the downloaded file
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Move the sub folder to the right place
            shutil.move(
                os.path.join(temp_dir, self.dataset.type.value, self.dataset.name),
                self._data_dir
            )

            # Delete the temp directory
            shutil.rmtree(temp_dir)

            # Delete the ZIP file
            os.remove(zip_file)

    def __iter__(self):
        self._entity_idx = 0
        return self

    def __next__(self):
        # Create the scaler
        scaler = MinMaxScaler()

        # Check if the entity index is valid
        try:
            self.dataset.entities[self._entity_idx]
        except IndexError:
            raise StopIteration

        # Create the train datasets
        train_dataset = TimeSeriesDataset(
            data_dir=self._data_dir,
            dataset=self.dataset,
            entity=self.dataset.entities[self._entity_idx],
            scaler=scaler,
            window_size=self._window_size,
            train=True
        )

        # Create the test datasets
        test_dataset = TimeSeriesDataset(
            data_dir=self._data_dir,
            dataset=self.dataset,
            entity=self.dataset.entities[self._entity_idx],
            scaler=scaler,
            window_size=self._window_size,
            train=False
        )

        # Create the train data loader
        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, num_workers=os.cpu_count() // 2)

        # Create the test data loader
        test_dataloader = DataLoader(test_dataset, batch_size=self._batch_size, num_workers=os.cpu_count() // 2)

        # Increment the entity index
        self._entity_idx += 1

        return train_dataloader, test_dataloader
