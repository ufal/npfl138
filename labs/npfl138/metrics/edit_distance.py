# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections.abc import Sequence
from typing import Self, Any

import torch

from ..metric import Metric


class EditDistance(torch.nn.Module, Metric):
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
