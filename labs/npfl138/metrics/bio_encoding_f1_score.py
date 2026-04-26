# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Self

import torch

from ..metric import Metric


class BIOEncodingF1Score(torch.nn.Module, Metric):
    """Metric for evaluating F1 score of BIO-encoded spans.

    The metric employs a simple heuristic to handle invalid sequences of BIO tags.
    Notably:

    - If there is an `I` tag without preceding `B/I` tag, it is considered a `B` tag.
    - If the type of an `I` tag does not match the type of the preceding tag, the type
      of this `I` tag is ignored (i.e., considered the same as the preceding tag type).
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
            span, offset, start = None, 0, None
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
