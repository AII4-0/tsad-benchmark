from typing import Dict, Any

import determined as det
import torch
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor


class EntityDeterminedLogger(Logger):
    """This class is an implementation of the Lightning's `Logger` that reports the logging to Determined AI."""

    def __init__(self) -> None:
        """Create an object of `EntityDeterminedLogger` class."""
        super().__init__()
        self._core_context = det.core.init()
        self.current_entity = 0
        self._metrics = {}

    @property
    def name(self) -> str:
        """
        Return the experiment name.

        :return: The experiment name.
        """
        return ""

    @property
    def version(self) -> str:
        """
        Return the experiment version.

        :return: The experiment version.
        """
        return ""

    @property
    def save_dir(self) -> str:
        """
        Return the root directory where experiment logs get saved.

        :return: The root directory where experiment logs get saved.
        """
        return "determined_logs"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Record hyperparameters.

        :param params: The hyperparameters to log.
        """

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Record metrics.

        :param metrics: The metrics to record.
        :param step: The step number.
        """
        # Pop the epoch from the metrics dictionary
        epoch = int(metrics.pop("epoch"))

        # Report differently depending on whether it is the loss or the parameters of the entity
        if "loss" in metrics:
            # Create a new dictionary with a metric name containing the entity number
            loss = {f"entity_{self.current_entity}_loss": metrics["loss"]}

            # Save the loss
            self._save_metrics(epoch, loss)
        else:
            # Save the entity metrics
            self._save_metrics(self.current_entity, metrics)

    def log_global_metrics(self, metrics: Dict[str, Tensor]) -> None:
        """
        Record global metrics.

        :param metrics: The global metrics to record.
        """
        # Convert tensors to floats
        metrics = {key: value.item() for key, value in metrics.items()}

        # Save the global metrics
        self._save_metrics(self.current_entity, metrics)

    def report_metrics(self) -> None:
        """Send metrics to Determined AI."""
        # Iterate over the saved metrics
        for step, metrics in self._metrics.items():
            # Report metrics to Determined
            self._core_context.train.report_training_metrics(step, metrics)

    def report_checkpoint(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Send checkpoints to Determined AI.

        :param state_dict: The model's state dictionary.
        """
        # Get the checkpoint store path
        with self._core_context.checkpoint.store_path({"steps_completed": self.current_entity}) as (path, uuid):
            # Save the model with `torch.save`
            torch.save(state_dict, path / "model.pt")

    def _save_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        # Create the step if it doesn't exist
        if step not in self._metrics:
            self._metrics[step] = {}

        # Merge data in the existing step dictionary
        self._metrics[step] = {**self._metrics[step], **metrics}
