from dataclasses import dataclass
from typing import List

from datasets.time_series_types import TimeSeriesTypes


@dataclass(init=True)
class DatasetDescriptor:
    """This class represents a dataset with all the necessary information."""

    name: str
    type: TimeSeriesTypes
    download_url: str
    dimension: int
    entities: List[str]
