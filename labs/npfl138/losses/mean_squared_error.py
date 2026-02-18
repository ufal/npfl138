# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from ..loss import Loss
from ..type_aliases import Reduction
from ..utils import broadcast_to_prefix, maybe_remove_one_singleton_dimension

mse_loss = torch.nn.functional.mse_loss


class MeanSquaredError(Loss):
    """Mean squared error loss implementation."""

    def __init__(self, *, reduction: Reduction = "mean") -> None:
        """Create the MeanSquaredError object with the specified reduction method.

        Parameters:
          reduction: The reduction method to apply to the computed loss.
        """

        self._reduction = reduction

    def __call__(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the mean squared error loss, optionally with sample weights.

        Parameters:
          y: The predicted outputs. Their shape either has to be exactly the same as `y_true` (no broadcasting),
            or can contain an additional single dimension of size 1.
          y_true: The ground-truth targets.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y_true`, and the loss for each sample is weighted accordingly.

        Returns:
          A tensor representing the computed loss. A scalar tensor if reduction is `"mean"` or `"sum"`;
            otherwise (if reduction is `"none"`), a tensor of the same shape as `y_true`.
        """
        y = maybe_remove_one_singleton_dimension(y, y_true)
        assert y.shape == y_true.shape, f"Shapes of y {y.shape} and y_true {y_true.shape} have to match " \
            "up to one singleton dim in y."

        if sample_weights is not None:
            sample_weights = broadcast_to_prefix(sample_weights, y_true.shape)

        return mse_loss(y, y_true, reduction=self._reduction, weight=sample_weights)
