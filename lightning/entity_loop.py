from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from pytorch_lightning.loops import Loop, FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor

from lightning.entity_determined_logger import EntityDeterminedLogger
from utils.metrics import f1_score_from_confusion_matrix, precision_from_confusion_matrix, recall_from_confusion_matrix


class EntityLoop(Loop):
    """This class is an implementation of the Lightning's `Loop` that loops over entities."""

    def __init__(self, num_entities: int) -> None:
        """
        Create an object of `EntityLoop` class.

        :param num_entities: The number of the entities.
        """
        super().__init__()
        # Define the public `fit_loop` attribute
        self.fit_loop = None

        # Define necessary private attributes
        self._num_entities = num_entities
        self._current_entity = 0
        self._lightning_module_state_dict = None

        # Define variable to store the global confusion matrix
        self._true_positives = torch.tensor(0.0)
        self._true_positives_adjusted = torch.tensor(0.0)
        self._true_negatives = torch.tensor(0.0)
        self._true_negatives_adjusted = torch.tensor(0.0)
        self._false_positives = torch.tensor(0.0)
        self._false_positives_adjusted = torch.tensor(0.0)
        self._false_negatives = torch.tensor(0.0)
        self._false_negatives_adjusted = torch.tensor(0.0)

    @property
    def done(self) -> bool:
        """
        Provide a condition to stop the loop.

        :return: True if it is done, otherwise False.
        """
        return self._current_entity >= self._num_entities

    def connect(self, fit_loop: FitLoop) -> None:
        """
        Connect a fitting loop to this loop.

        :param fit_loop: The fitting loop.
        """
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self) -> None:
        """Store the original weights of the models."""
        self._lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self) -> None:
        """Call `setup_entity` function from the `EntityDataModule` instance."""
        # Set up the data for the current entity
        # noinspection PyUnresolvedReferences
        self.trainer.datamodule.setup_entity(self._current_entity)

        # Update the current entity in the Determined AI logger (only if used)
        determined_logger = self._determined_logger()
        if determined_logger:
            determined_logger.current_entity = self._current_entity

    def advance(self) -> None:
        """Run a fitting and testing on the current entity."""
        # Reset the fitting stage
        self._reset_fitting()

        # Run the original fit loop
        self.fit_loop.run()

        # Reset the testing stage
        self._reset_testing()

        # Run the test loop
        outputs = self._run_test_loop()

        # Update the global confusion matrix
        self._true_positives += outputs["true_positives"].cpu()
        self._true_positives_adjusted += outputs["true_positives_adjusted"].cpu()
        self._true_negatives += outputs["true_negatives"].cpu()
        self._true_negatives_adjusted += outputs["true_negatives_adjusted"].cpu()
        self._false_positives += outputs["false_positives"].cpu()
        self._false_positives_adjusted += outputs["false_positives_adjusted"].cpu()
        self._false_negatives += outputs["false_negatives"].cpu()
        self._false_negatives_adjusted += outputs["false_negatives_adjusted"].cpu()

        # Update the current entity by increment it
        self._current_entity += 1

    def on_advance_end(self) -> None:
        """Save the weights of the current entity. Then, restore the original weights and optimizers."""
        # Report the checkpoint to Determined AI (only if used)
        determined_logger = self._determined_logger()
        if determined_logger:
            determined_logger.report_checkpoint(self.trainer.lightning_module.state_dict())

        # Restore the original weights and optimizers
        self.trainer.lightning_module.load_state_dict(self._lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Compute the global metrics."""
        # Compute global metrics
        global_metrics = {
            "global_true_positives": self._true_positives,
            "global_true_positives_adjusted": self._true_positives_adjusted,
            "global_true_negatives": self._true_negatives,
            "global_true_negatives_adjusted": self._true_negatives_adjusted,
            "global_false_positives": self._false_positives,
            "global_false_positives_adjusted": self._false_positives_adjusted,
            "global_false_negatives": self._false_negatives,
            "global_false_negatives_adjusted": self._false_negatives_adjusted,
            "global_precision": precision_from_confusion_matrix(
                self._true_positives,
                self._true_negatives,
                self._false_positives,
                self._false_negatives
            ),
            "global_precision_adjusted": precision_from_confusion_matrix(
                self._true_positives_adjusted,
                self._true_negatives_adjusted,
                self._false_positives_adjusted,
                self._false_negatives_adjusted
            ),
            "global_recall": recall_from_confusion_matrix(
                self._true_positives,
                self._true_negatives,
                self._false_positives,
                self._false_negatives
            ),
            "global_recall_adjusted": recall_from_confusion_matrix(
                self._true_positives_adjusted,
                self._true_negatives_adjusted,
                self._false_positives_adjusted,
                self._false_negatives_adjusted
            ),
            "global_f1": f1_score_from_confusion_matrix(
                self._true_positives,
                self._true_negatives,
                self._false_positives,
                self._false_negatives
            ),
            "global_f1_adjusted": f1_score_from_confusion_matrix(
                self._true_positives_adjusted,
                self._true_negatives_adjusted,
                self._false_positives_adjusted,
                self._false_negatives_adjusted
            )
        }

        # Print the global metrics
        self._print_global_metrics(global_metrics)

        # Add global metrics and report all metrics to Determined AI (only if used)
        determined_logger = self._determined_logger()
        if determined_logger:
            determined_logger.log_global_metrics(global_metrics)
            determined_logger.report_metrics()

    def _reset_fitting(self) -> None:
        """Reset the fitting stage."""
        self.trainer.reset_train_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        """Reset the testing stage."""
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def _run_test_loop(self) -> Dict[str, Tensor]:
        """
        Run the test loop.

        Note: the test loop normally expects the models to be the pure LightningModule, but since we are running the
              test loop during fitting, we need to temporarily unpack the wrapped module.

        :return: The test output.
        """
        # Unpack
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module

        # Set the model to eval mode
        self.trainer.strategy.model.eval()
        torch.set_grad_enabled(False)

        # Run the test
        outputs = self.trainer.test_loop.run()

        # Set the model to train mode
        self.trainer.strategy.model.train()
        torch.set_grad_enabled(True)

        # Repack
        self.trainer.strategy.model = wrapped_model

        return outputs[0]

    def _determined_logger(self) -> Optional[EntityDeterminedLogger]:
        if isinstance(self.trainer.logger, EntityDeterminedLogger):
            return self.trainer.logger
        return None

    @staticmethod
    def _print_global_metrics(metrics: Dict[str, Tensor]) -> None:
        # Compute the max lengths of the key and the value
        max_len_key = max([len(k) for k in metrics.keys()])
        max_len_value = max([len(str(v.numpy())) for v in metrics.values()])

        # Compute the width
        width = 3 * 5 + max_len_key + max_len_value

        # Header
        left_pad = (width - 14) // 2
        print("─" * width)
        print((" " * left_pad) + "Global metrics")
        print("─" * width)

        # Rows
        for k, v in metrics.items():
            v = str(v.numpy())
            left_pad = (max_len_key - len(k)) // 2
            middle_pad = max_len_key - len(k) - left_pad + 5 + (max_len_value - len(v)) // 2
            print("     " + (" " * left_pad) + k + (" " * middle_pad) + v)

        # Footer
        print("─" * width)

    def __getattr__(self, key: str) -> Any:  # noqa: ANN401
        """
        Return the value of the specified attribute.

        :param key: The key of the attribute.
        :return: The attribute corresponding to the key.
        """
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Set the state.

        :param state: The state dictionary.
        """
        self.__dict__.update(state)
