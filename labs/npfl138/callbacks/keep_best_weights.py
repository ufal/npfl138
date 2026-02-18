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


class KeepBestWeights(Callback):
    """A callback that keeps the best model weights in memory."""

    def __init__(
        self,
        metric: str,
        mode: Literal["max", "min"] = "max",
        device: str | None = None,
    ) -> None:
        """Create the KeepBestWeights callback.

        Parameters:
          metric: The metric name from `logs` dictionary to monitor.
          mode: One of `"max"` or `"min"`, indicating whether the monitored metric should be maximized
            or minimized.
          device: The device where the weights will be stored. If `None`, the weights will be stored
            on the same device as the model.
        """
        assert mode in ("max", "min"), "mode must be one of 'max' or 'min'"

        self._metric = metric
        self._mode = mode
        self._device = device

        self.best_state_dict = None
        self.best_value = None

    best_state_dict: dict | None
    """The state dictionary containing the copies of best weights encountered so far."""

    best_value: float | None = None
    """The best metric value seen so far."""

    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> None:
        if (self.best_value is None
                or (self._mode == "max" and logs[self._metric] > self.best_value)
                or (self._mode == "min" and logs[self._metric] < self.best_value)):
            self.best_value = logs[self._metric]
            self.best_state_dict = {k: v.to(device=self._device, copy=True)
                                    for k, v in module.state_dict().items()}
