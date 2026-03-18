# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Sequence
import math
from typing import Self

import torch

from ..metric import Metric


class MaskIoU(torch.nn.Module, Metric):
    """An implementation of mean IoU metric computed on binary masks."""
    def __init__(self, mask_shape: Sequence[int], from_logits: bool = False) -> None:
        """Construct a new MaskIoU metric.

        Parameters:
          mask_shape: The shape of the input masks as (H, W).
          from_logits: If `True`, the predictions are expected to be logits; otherwise, they
            are probabilities (the default). However, the target masks must always be probabilities.
        """
        super().__init__()
        self.register_buffer("iou", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64), persistent=False)
        self._mask_size = math.prod(mask_shape)
        self._prediction_threshold = 0.0 if from_logits else 0.5

    def reset(self) -> Self:
        """Reset the metric to its initial state.

        Returns:
          self
        """
        self.iou.zero_()
        self.count.zero_()
        return self

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Self:
        """Update the metric with new predictions and targets.

        Returns:
          self
        """
        y_pred_mask = (y_pred.detach() >= self._prediction_threshold).reshape([-1, self._mask_size])
        y_true_mask = (y_true.detach() >= 0.5).reshape([-1, self._mask_size])

        intersection = torch.logical_and(y_pred_mask, y_true_mask).float().sum(dim=1)
        union = torch.logical_or(y_pred_mask, y_true_mask).float().sum(dim=1)
        iou = torch.where(union == 0, 1., intersection / union)

        self.iou += iou.sum()
        self.count += iou.shape[0]
        return self

    def compute(self) -> torch.Tensor:
        """Compute the mean IoU."""
        return self.iou / self.count
