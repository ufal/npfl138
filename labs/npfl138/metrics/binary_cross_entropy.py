# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Self

import torch

from .mean import Mean
from .. import losses


class BinaryCrossEntropy(Mean):
    """Binary cross-entropy metric implementation."""

    def __init__(
        self, *, label_smoothing: float = 0.0, probs: bool = False, device: torch.device | None = None,
    ) -> None:
        """Create the BinaryCrossEntropy metric object.

        Parameters:
          label_smoothing: A float in [0.0, 1.0] specifying the label smoothing factor.
            If greater than 0.0, the used ground-truth targets are computed as a mixture
            of the original targets and uniform distribution with weight `1 - label_smoothing`.
          probs: If `False`, the predictions are assumed to be logits; if `True`, the
            predictions are assumed to be probabilities. Note that gold targets are
            always expected to be probabilities.
        """
        super().__init__(device)
        self._bce_loss = losses.BinaryCrossEntropy(label_smoothing=label_smoothing, probs=probs, reduction="none")

    @torch.no_grad
    def update(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> Self:
        """Update the accumulated binary cross-entropy by introducing new values.

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
        return super().update(self._bce_loss(y, y_true), sample_weights=sample_weights)
