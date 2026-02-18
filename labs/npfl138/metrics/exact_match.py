# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Iterable
from typing import Any, Self

import torch

from .mean import Mean


class ExactMatch(Mean):
    """Exact match metric implementation.

    The elements to compare can be either tensors or generic iterables. When tensors are used,
    the `element_dims` parameter can be specified to indicate which dimensions of the tensors
    form an element for comparison; when iterables are used, the input one-dimensional sequences are
    compared directly.
    """

    def __init__(self, element_dims: int | tuple[int] = (), device: torch.device | None = None) -> None:
        """Create the ExactMatch metric object.

        Parameters:
          element_dims: If the values to compare are tensors, this parameter
            can be used to specify which dimensions of the tensors form
            an element for comparison.
        """
        super().__init__(device)
        if isinstance(element_dims, int):
            self._element_dims = (element_dims,)
        elif isinstance(element_dims, (tuple, list)):
            self._element_dims = tuple(element_dims)
        else:
            raise TypeError("The element_dims argument must be an int or a tuple of ints.")

    @torch.no_grad
    def update(
        self,
        y: torch.Tensor | Iterable[Any],
        y_true: torch.Tensor | Iterable[Any],
        sample_weights: torch.Tensor | None = None,
    ) -> Self:
        """Update the exact match by comparing the given values.

        The inputs can be either both tensors or both iterables. When they are both tensors,
        the `element_dims` parameter can be used to specify which dimensions of the tensors
        form an element for comparison; when they are both iterables, the elements are
        compared directly.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: A tensor or an iterable of predicted values of the same shape as `y_true`.
          y_true: A tensor or an iterable of ground-truth targets of the same shape as `y`.
          sample_weights: Optional sample weights. Their shape must be broadcastable to a
            prefix of the shape of `y` (with `element_dims` dimensions removed, if specified).

        Returns:
          self
        """
        if isinstance(y, torch.Tensor) and isinstance(y_true, torch.Tensor):
            assert y.shape == y_true.shape, "The y and y_true tensors must have the same shape."

            equals = (y == y_true)
            if self._element_dims:
                equals = torch.all(equals, dim=self._element_dims)
        else:
            assert not self._element_dims, "Nonempty element_dims can only be used with tensor inputs."
            equals = torch.tensor([pred == true for pred, true in zip(y, y_true, strict=True)], dtype=torch.float32)

        return super().update(equals, sample_weights=sample_weights)
