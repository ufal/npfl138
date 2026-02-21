# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Protocol

import torch


class Metric(Protocol):
    """An abstract metric interface.

    Note:
      Metrics are expected to be instances of [torch.nn.Module][] and are stored
        in a [torch.nn.ModuleDict][].
    """

    def update(self, y: torch.Tensor, y_true: torch.Tensor) -> Any:
        """Update the internal state of the metric with new predictions and gold targets.

        Parameters:
          y: The predicted outputs.
          y_true: The ground-truth targets.

        Returns:
          anything; `npfl138` metrics return `Self`, but any return value is allowed in the generic
            interface (`torchmetrics` metrics return `None`, for example).
        """
        ...

    def compute(self) -> torch.Tensor:
        """Compute the accumulated metric value.

        Returns:
          A (usually scalar) tensor representing the accumulated metric value.
        """
        ...

    def reset(self) -> Any:
        """Reset the internal state of the metric.

        Returns:
          anything; `npfl138` metrics return `Self`, but any return value is allowed in the generic
            interface (`torchmetrics` metrics return `None`, for example).
        """
        ...
