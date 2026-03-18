# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Literal, TYPE_CHECKING

from ..callback import Callback, StopTraining, STOP_TRAINING
from ..type_aliases import Logs
if TYPE_CHECKING:
    from ..trainable_module import TrainableModule


class SaveBestWeights(Callback):
    """A callback that saves best model weights to a file."""

    def __init__(
        self,
        path: str,
        metric: str,
        mode: Literal["max", "min"] = "max",
        optimizer_path: str | None = None,
        patience: int | None = None,
    ) -> None:
        """Create the SaveBestWeights callback.

        Parameters:
          path: A path where weights will be saved using the [npfl138.TrainableModule.save_weights][]
            method after each epoch. Note that you can use templates like `{logdir}` and `{epoch[:formatting]}`.
          metric: The metric name from `logs` dictionary to monitor.
          mode: One of `"max"` or `"min"`, indicating whether the monitored metric should be maximized
            or minimized.
          optimizer_path: An optional path passed to [npfl138.TrainableModule.save_weights][] to
            save also the optimizer state; it is relative to `path`.
          patience: When `patience` is not `None`, the callback stops the training if the monitored
            metric does not improve for `patience` consecutive epochs.
        """
        assert mode in ("max", "min"), "mode must be one of 'max' or 'min'"

        self._path = path
        self._metric = metric
        self._mode = mode
        self._optimizer_path = optimizer_path
        self._patience = patience
        self._epochs_without_improvement = 0

        self.best_value = None

    best_value: float | None
    """The best metric value seen so far."""

    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> StopTraining | None:
        if (self.best_value is None
                or (self._mode == "max" and logs[self._metric] > self.best_value)
                or (self._mode == "min" and logs[self._metric] < self.best_value)):
            self.best_value = logs[self._metric]
            module.save_weights(self._path, optimizer_path=self._optimizer_path)
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._patience is not None and self._epochs_without_improvement >= self._patience:
            return STOP_TRAINING
