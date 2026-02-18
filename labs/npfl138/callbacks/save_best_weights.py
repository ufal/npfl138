# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Literal, TYPE_CHECKING

from ..callback import Callback
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
        """
        assert mode in ("max", "min"), "mode must be one of 'max' or 'min'"

        self._path = path
        self._metric = metric
        self._mode = mode
        self._optimizer_path = optimizer_path

        self.best_metric_value = None

    best_value: float | None = None
    """The best metric value seen so far."""

    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> None:
        if (self.best_value is None
                or (self._mode == "max" and logs[self._metric] > self.best_value)
                or (self._mode == "min" and logs[self._metric] < self.best_value)):
            self.best_value = logs[self._metric]

            module.save_weights(self._path, optimizer_path=self._optimizer_path)
