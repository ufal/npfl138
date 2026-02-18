# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Self

import torch

from .mean import Mean
from ..utils import maybe_remove_one_singleton_dimension


class BinaryAccuracy(Mean):
    """Binary classification accuracy metric.

    The predictions are assumed to be logits or probabilities predicted by a model,
    while the ground-truth targets are binary (0 or 1) values. In both cases, the predicted
    class is considered to be the one with larger probability.
    """

    def __init__(self, *, probs: bool = False, device: torch.device | None = None) -> None:
        """Create the BinaryAccuracy metric object.

        Parameters:
          probs: If `False`, the predictions are assumed to be logits; if `True`, the
            predictions are assumed to be probabilities. Note that gold targets are
            always expected to be probabilities.
        """
        super().__init__(device)
        self._probs = probs

    @torch.no_grad
    def update(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> Self:
        """Update the accumulated binary accuracy using new predictions and gold labels.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: The predicted outputs. Their shape either has to be exactly the same as `y_true` (no broadcasting),
            or can contain an additional single dimension of size 1. We consider the more probable class
            to be predicted.
          y_true: The ground-truth targets; they are rounded to 0 or 1 to obtain binary labels.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y_true`, and the loss for each sample is weighted accordingly.

        Returns:
          self
        """
        y = maybe_remove_one_singleton_dimension(y, y_true)
        assert y.shape == y_true.shape, f"Shapes of y {y.shape} and y_true {y_true.shape} have to match " \
            "up to one singleton dim in y."

        y = (y > (0.5 if self._probs else 0.0))
        y_true = (y_true > 0.5)

        return super().update(y == y_true, sample_weights=sample_weights)
