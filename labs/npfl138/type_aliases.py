# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Types and type aliases used by NPFL138."""
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import torch


AnyArray: TypeAlias = torch.Tensor | np.ndarray | list | tuple
"""A type alias for any array-like structure.

PyTorch tensors, NumPy arrays, lists, and tuples are supported.
"""


DataFormat: TypeAlias = Literal["HWC", "CHW"]
"""A type alias for image data format description."""


class HasCompute(Protocol):
    """A protocol for objects that have a `compute` method of the following type:

    ```python
    def compute(self) -> float | torch.Tensor | np.ndarray:
        ...
    ```
    """
    def compute(self) -> float | torch.Tensor | np.ndarray:
        """Compute the value of the object."""
        ...


Logs: TypeAlias = dict[str, float | torch.Tensor | np.ndarray | HasCompute]
"""A dictionary of logs, with keys being the log names and values being the log values.

When the logs are returned by a [npfl138.TrainableModule][] or passed to a [npfl138.Callback][],
they are always evaluated to just float values.
"""

Reduction: TypeAlias = Literal["mean", "sum", "none"]
"""A type alias for reduction methods used in losses and metrics."""


Tensor: TypeAlias = torch.Tensor | torch.nn.utils.rnn.PackedSequence
"""A type alias for a single tensor or a packed sequence of tensors."""


TensorOrTensors: TypeAlias = Tensor | tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor] | Any
"""A type alias for a single tensor or a tensor structure.

While a tensor or a sequence of them is the most common, any type is allowed
here to accomodate nested or completely custom data structures.
"""
