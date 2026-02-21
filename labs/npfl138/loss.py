# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Protocol

import torch


class Loss(Protocol):
    """An abstract loss function interface."""

    def __call__(self, y: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss of the given predictions and gold targets.

        Parameters:
          y: The predicted outputs.
          y_true: The ground-truth targets.

        Returns:
          A tensor representing the computed loss.
        """
        ...
