# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Self

import torch

from .mean import Mean
from ..utils import broadcast_to_prefix


class CategoricalAccuracy(Mean):
    """Categorical classification accuracy metric.

    The predictions are assumed to be logits or probabilities predicted by a model,
    while the ground-truth targets are either class indices (sparse format) or whole
    distributions (dense format). In both cases, the predicted class is considered
    to be the one with the highest probability.
    """

    def __init__(
        self, dim: int = 1, *, ignore_index: int = -100, probs: bool = False, device: torch.device | None = None,
    ) -> None:
        """Create the CategoricalAccuracy metric object.

        Parameters:
          dim: If the input has 2 or more dimensions, this value specifies the dimension along which
            the classes are defined. The default is the same behavior as [torch.nn.CrossEntropyLoss][].
          ignore_index: An optional target class value that is ignored during metric computation (equivalent
            to zeroing out sample weights for the corresponding samples). The default of -100 is the same
            as in [torch.nn.CrossEntropyLoss][].
          probs: For consistency with other categorical losses and metrics, we include this parameter
            describing whether the predictions are logits or probabilities. However, to predict the
            most probable class, it does not matter whether logits or probabilities are used.
        """
        super().__init__(device)
        self._dim = dim
        self._ignore_index = ignore_index

    @torch.no_grad
    def update(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> Self:
        """Update the accumulated categorical accuracy using new predictions and gold labels.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: The predicted outputs, either logits or probabilities (depending on the `probs` parameter).
            If they have 2 or more dimensions, the class dimension is specified by the `dim` parameter.
            We consider the class with the highest probability to be predicted.
          y_true: The ground-truth targets in two possible formats:

            - The gold targets might be "sparse" class indices. In this case, their shape has to be
              exactly the same as `y` with the class dimension removed.
            - The gold targets might be full "dense" probability distributions. In this case, their
              shape has to be exactly the same as `y` and we consider the class with the highest probability
              to be the gold class.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y` with the class dimension removed, and the loss for each sample
            is weighted accordingly.

        Returns:
          self
        """
        y_shape, y_true_shape = y.shape, y_true.shape
        dim = self._dim % len(y_shape) if len(y_shape) > 1 else 0
        y_wo_class_dim_shape = y_shape[:dim] + y_shape[dim + 1:]

        dense = len(y_true_shape) == len(y_shape)
        if dense:
            assert y_true_shape == y_shape, "In dense format, y_true must have the same shape as y."
            y_true = torch.argmax(y_true, dim=dim)
        else:
            assert y_true_shape == y_wo_class_dim_shape, \
                "In sparse format, y_true must have the same shape as y with the class dimension removed."
            y_true_dtype = y_true.dtype
            assert not y_true_dtype.is_floating_point and not y_true_dtype.is_complex, \
                "In sparse format, y_true must contain class indices."
            if y_true_dtype != torch.int64 and y_true_dtype != torch.uint8:
                y_true = y_true.long()

        y = torch.argmax(y, dim=dim)

        ignore_index_weights = (y_true != self._ignore_index).to(dtype=torch.float32)
        if sample_weights is None:
            sample_weights = ignore_index_weights
        else:
            sample_weights = broadcast_to_prefix(sample_weights, y_wo_class_dim_shape)
            sample_weights = sample_weights.to(dtype=torch.float32) * ignore_index_weights

        return super().update(y == y_true, sample_weights=sample_weights)
