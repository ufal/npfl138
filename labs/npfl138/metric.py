# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from typing import Self

import torch

from .type_aliases import TensorOrTensors


class Metric(torch.nn.Module, ABC):
    """An abstract metric interface."""

    @abstractmethod
    def update(
        self, y: TensorOrTensors, y_true: TensorOrTensors | None = None, sample_weights: TensorOrTensors | None = None,
    ) -> Self:
        """Update the internal state of the metric with new predictions and possibly gold targets.

        Optional sample weights might be provided if supported by the metric.

        Parameters:
          y: The predicted outputs.
          y_true: Optional ground-truth targets.
          sample_weights: Optional sample weights.

        Returns:
          self
        """
        ...

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute the accumulated metric value.

        Returns:
          A (usually scalar) tensor representing the accumulated metric value.
        """
        ...

    @abstractmethod
    def reset(self) -> Self:
        """Reset the internal state of the metric.

        Returns:
          self
        """
        ...
