from typing import Optional

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """This class is tqdm progress bar for download."""

    def __init__(self, dataset_name: str) -> None:
        """
        Create an object of the `DownloadProgressBar` class.

        :param dataset_name: The name of the dataset that will be downloaded.
        """
        super().__init__(unit="B", unit_scale=True, miniters=1, desc="Download " + dataset_name + " datasets")

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        """
        Update the progress bar.

        :param b: Number of blocks transferred so far.
        :param bsize: Size of each block (in tqdm units).
        :param tsize: Total size (in tqdm units).
        :return: True if a `display()` was triggered, otherwise False.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
