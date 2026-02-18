# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import re
from typing import Any

import torch

from .type_aliases import Logs


def broadcast_to_prefix(tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Broadcast the given tensor to the given shape from the left (prefix).

    The broadcasting is performed by unsqueezing the tensor's dimensions until
    it has the same number of dimensions as the given shape, and then expanding it.

    Parameters:
      tensor: The tensor to broadcast.
      shape: The shape to broadcast the tensor to (from the left).

    Returns:
      The broadcasted tensor.
    """
    while tensor.dim() < len(shape):
        tensor = tensor.unsqueeze(dim=-1)
    if tensor.shape != shape:
        tensor = tensor.expand(shape)
    return tensor


def compute_logs(logs: Logs) -> dict[str, float]:
    """Evaluate the given logs dictionary to floats, calling `compute` where needed.

    Parameters:
      logs: The logs dictionary which can contain float values, PyTorch and Numpy tensors,
        or [HasCompute][]-compliant objects providing a `compute` method.

    Returns:
      logs: The same dictionary as on input, but with all values evaluated to floats.
    """
    for k, v in logs.items():
        if not isinstance(v, float):
            v = v.compute() if hasattr(v, "compute") else v
            logs[k] = float(v.item() if hasattr(v, "item") else v)
    return logs


def maybe_remove_one_singleton_dimension(y: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Possibly remove one singleton dimension from the given tensor `y` to match the shape of `y_true`.

    If `y` has one more dimension than `y_true` and that extra dimension is a dimension of size 1,
    it will remove that dimension from `y`. Otherwise, `y` is returned unchanged.

    Parameters:
      y: The predicted outputs.
      y_true: The ground-truth targets.

    Returns:
      The tensor `y` with a surplus singleton dimension possibly removed.
    """
    y_shape, y_true_shape = y.shape, y_true.shape

    if len(y_shape) == len(y_true_shape) + 1:
        singleton_dim = 0
        while singleton_dim < len(y_true_shape) and y_shape[singleton_dim] == y_true_shape[singleton_dim]:
            singleton_dim += 1
        if y_shape[singleton_dim] == 1:
            y = y.squeeze(dim=singleton_dim)

    return y


def fill_and_standardize_path(path: str, **kwargs: Any) -> str:
    """Fill placeholders in the path and standardize path separators.

    The template placeholders `{key}` in the path are replaced with the corresponding values
    from `kwargs` using `str.format`, and the both slashes and backslashes are replaced
    with the current OS path separator.

    Parameters:
      path: The path template with placeholders.
      **kwargs: The keyword arguments to fill the placeholders in the path.

    Returns:
      The standardized path with filled placeholders and OS-specific separators.
    """
    filled_path = path.format(**kwargs)
    standardized_path = filled_path.replace("\\", os.path.sep).replace("/", os.path.sep)
    return standardized_path


def sanitize_path(path: str) -> str:
    """Sanitize the given path by replacing path-unfriendly characters with underscores.

    Parameters:
      path: The input path to sanitize.

    Returns:
      The sanitized path.
    """
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", path)


tuple_list: tuple[type, type] = (tuple, list)
