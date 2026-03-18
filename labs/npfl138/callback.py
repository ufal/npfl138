# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Protocol, Self, TYPE_CHECKING

from .type_aliases import Logs
if TYPE_CHECKING:
    from .trainable_module import TrainableModule


class StopTraining:
    """A sentinel type whose any instance can be returned by a callback to stop training."""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


STOP_TRAINING = StopTraining()
"""A sentinel value that can be returned by a callback to stop training."""


class Callback(Protocol):
    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs, /) -> StopTraining | None:
        """Represents a callback called after every training epoch.

        If the callback returns [npfl138.STOP_TRAINING][], the training stops.

        Parameters:
          module: the module being trained
          epoch: the current epoch number (starting from one)
          logs: a dictionary of current logs, with all values being evaluated to float;
            newly computed metrics can be added here

        **Returns:**

          - [`npfl138.STOP_TRAINING`][npfl138.STOP_TRAINING] to stop the training,
          - [`None`][None] to continue.
        """
        ...
