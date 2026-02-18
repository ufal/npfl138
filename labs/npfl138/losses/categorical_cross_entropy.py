# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from ..loss import Loss
from ..type_aliases import Reduction
from ..utils import broadcast_to_prefix

cross_entropy_loss = torch.nn.functional.cross_entropy


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy loss implementation."""

    def __init__(
        self,
        dim: int = 1,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        probs: bool = False,
        reduction: Reduction = "mean",
    ) -> None:
        """Create the CategoricalCrossEntropy loss object with the specified reduction method.

        Parameters:
          dim: If the input has 2 or more dimensions, this value specifies the dimension along which
            the classes are defined. The default is the same behavior as [torch.nn.CrossEntropyLoss][].
          ignore_index: An optional target class value that is ignored during loss computation (equivalent
            to zeroing out sample weights for the corresponding samples). Only applicable for sparse targets;
            when dense targets are used, the default of -100 cannot be overwritten and this parameter is
            ignored. This is the same behavior as [torch.nn.CrossEntropyLoss][].
          label_smoothing: A float in [0.0, 1.0] specifying the label smoothing factor.
            If greater than 0.0, the used ground-truth targets are computed as a mixture
            of the original targets and uniform distribution with weight `1 - label_smoothing`.
          probs: If `False`, the predictions are assumed to be logits; if `True`, the
            predictions are assumed to be probabilities. Note that gold targets are
            always expected to be probabilities in the dense format.
          reduction: The reduction method to apply to the computed loss.
        """

        self._dim = dim
        self._ignore_index = ignore_index
        self._label_smoothing = label_smoothing
        self._probs = probs
        self._reduction = reduction

    def __call__(
        self, y: torch.Tensor, y_true: torch.Tensor, sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the categorical cross-entropy loss, optionally with sample weights.

        Parameters:
          y: The predicted outputs, either logits or probabilities (depending on the `probs` parameter).
            If they have 2 or more dimensions, the class dimension is specified by the `dim` parameter.
          y_true: The ground-truth targets in two possible formats:

            - The gold targets might be "sparse" class indices. In this case, their shape has to be
              exactly the same as `y` with the class dimension removed.
            - The gold targets might be full "dense" probability distributions. In this case, their
              shape has to be exactly the same as `y`.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y` with the class dimension removed, and the loss for each sample
            is weighted accordingly.

        Returns:
          A tensor representing the computed loss. A scalar tensor if reduction is `"mean"` or `"sum"`;
            otherwise (if reduction is `"none"`), a tensor of the same shape as `y` without the class dimension.
        """
        y_shape, y_true_shape = y.shape, y_true.shape
        dim = self._dim % len(y_shape) if len(y_shape) > 1 else 0
        y_wo_class_dim_shape = y_shape[:dim] + y_shape[dim + 1:]

        dense = len(y_true_shape) == len(y_shape)
        if dense:
            assert y_true_shape == y_shape, "In dense format, y_true must have the same shape as y."
            assert self._ignore_index == -100, "When ignore_index is set, y_true cannot be in dense format."
        else:
            assert y_true_shape == y_wo_class_dim_shape, \
                "In sparse format, y_true must have the same shape as y with the class dimension removed."
            y_true_dtype = y_true.dtype
            assert not y_true_dtype.is_floating_point and not y_true_dtype.is_complex, \
                "In sparse format, y_true must contain class indices."
            if y_true_dtype != torch.int64 and y_true_dtype != torch.uint8:
                y_true = y_true.long()

        if self._probs:
            y = y.clamp(min=1e-7, max=1.0 - 1e-7).log()

        if len(y_shape) > 1 and dim != 1:
            y = y.movedim(dim, 1)
            if dense:
                y_true = y_true.movedim(dim, 1)

        if sample_weights is None:
            return cross_entropy_loss(y, y_true, ignore_index=self._ignore_index,
                                      label_smoothing=self._label_smoothing, reduction=self._reduction)
        else:
            sample_weights = broadcast_to_prefix(sample_weights, y_wo_class_dim_shape)

            losses = sample_weights * cross_entropy_loss(y, y_true, ignore_index=self._ignore_index,
                                                         label_smoothing=self._label_smoothing, reduction="none")
            if self._reduction == "none":
                return losses
            elif self._reduction == "sum":
                return losses.sum()
            else:  # self._reduction == "mean"
                return losses.sum() / sample_weights.sum()
