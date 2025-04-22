# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from typing import Any, Sequence
Self = Any  # For compatibility with Python <3.11 that does not support Self

import torch


class BIOEncodingF1Score(torch.nn.Module):
    """Metric for evaluating F1 score of BIO-encoded spans.

    The metric employs a simple heuristic to handle invalid sequences of BIO tags.
    Notably:

    - If there is an `I` tag without preceding `B/I` tag, it is considered a `B` tag.
    - If the type of an `I` tag does not match the type of the preceding tag, the type
      of this `I` tag is ignored (i.e., considered the same as the preceeding tag type).
    """
    def __init__(self, labels: list[str], ignore_index: int) -> None:
        """Construct a new BIOEncodingF1Score metric.

        Parameters:
          labels: The list of BIO-encoded labels.
          ignore_index: The gold index to ignore when computing the F1 score.
        """
        super().__init__()
        self.register_buffer("tp", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("fp", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("fn", torch.tensor(0, dtype=torch.int64), persistent=False)
        self._labels = labels
        self._ignore_index = ignore_index

    def reset(self) -> Self:
        """Reset the metric to its initial state.

        Returns:
          self
        """
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()
        return self

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> Self:
        """Update the metric with new predictions and targets.

        Returns:
          self
        """
        true = torch.nn.functional.pad(true, (0, 1), value=self._ignore_index).view(-1)
        pred = torch.nn.functional.pad(pred, (0, 1), value=self._ignore_index).view(-1)
        spans_pred, spans_true = set(), set()
        for spans, tags in [(spans_true, true), (spans_pred, pred)]:
            span, offset = None, 0
            for tag in tags:
                label = self._labels[tag] if tag != self._ignore_index else "O"
                if span and label.startswith(("O", "B")):
                    spans.add((start, offset, span))
                    span = None
                if not span and label.startswith(("B", "I")):
                    span, start = label[1:], offset
                if tag != self._ignore_index:
                    offset += 1
        self.tp.add_(len(spans_pred & spans_true))
        self.fp.add_(len(spans_pred - spans_true))
        self.fn.add_(len(spans_true - spans_pred))
        return self

    def compute(self) -> torch.Tensor:
        """Compute the F1 score."""
        return 2 * self.tp / torch.max(2 * self.tp + self.fp + self.fn, torch.ones_like(self.tp))


class EditDistance(torch.nn.Module):
    """An implementation of mean edit distance metric."""

    def __init__(self, ignore_index: int | None = None) -> None:
        """Construct a new EditDistance metric.

        Parameters:
          ignore_index: If not None, the gold index to ignore when computing the edit distance.
            The default is None, which means no index is ignored.
        """
        super().__init__()
        self._ignore_index = ignore_index
        self.register_buffer("edit_distances", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64), persistent=False)

    def reset(self) -> Self:
        """Reset the metric to its initial state.

        Returns:
          self
        """
        self.edit_distances.zero_()
        self.count.zero_()
        return self

    def update(self, y_preds: Sequence[Sequence[Any]], y_trues: Sequence[Sequence[Any]]) -> Self:
        """Update the metric with new predictions and targets.

        Returns:
          self
        """
        import torchaudio

        for y_pred, y_true in zip(y_preds, y_trues):
            if self._ignore_index is not None:
                y_true = [y for y in y_true if y != self._ignore_index]
                y_pred = [y for y in y_pred if y != self._ignore_index]
            self.edit_distances += torchaudio.functional.edit_distance(y_pred, y_true) / (len(y_true) or 1)
            self.count += 1
        return self

    def compute(self) -> torch.Tensor:
        """Compute the mean edit distance."""
        return self.edit_distances / self.count


class MaskIoU(torch.nn.Module):
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
