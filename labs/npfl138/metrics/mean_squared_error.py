# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Self

import torch

from .mean import Mean
from .. import losses


class MeanSquaredError(Mean):
    """Mean squared error metric implementation."""

    def __init__(self, device: torch.device | None = None) -> None:
        """Create the MeanSquaredError metric object."""
        super().__init__(device)
        self._mse_loss = losses.MeanSquaredError(reduction="none")

    @torch.no_grad
    def update(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> Self:
        """Update the accumulated mean squared error by introducing new values.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: The predicted outputs. Their shape either has to be exactly the same as `y_true` (no broadcasting),
            or can contain an additional single dimension of size 1.
          y_true: The ground-truth targets.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y_true`, and the loss for each sample is weighted accordingly.

        Returns:
          self
        """
        return super().update(self._mse_loss(y, y_true), sample_weights=sample_weights)
