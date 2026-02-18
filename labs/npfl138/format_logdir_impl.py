# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import datetime
import inspect
import os
import re
from typing import Any

from .utils import fill_and_standardize_path, sanitize_path


def format_logdir(logdir_template: str, **kwargs: Any) -> str:
    """Format the log directory path by filling in placeholders.

    The `logdir_template` is formatted using `str.format`, where the `{key}` placeholders
    are replaced by the corresponding values from `kwargs`. Importantly, several
    placeholders are always provided automatically:

    - `{config}`: A comma-separated list of `key=value` pairs for sorted key-value items in `kwargs`.
        The keys are abbreviated to their first character per segment (with segments separated by
        hyphens or underscores). The maximum length of the placeholder is limited to 200 characters;
        if exceeded, the longest entries are truncated with ellipses (`...`) to fit within the limit.
    - `{file}`: The base name of the script file (without extension) that called this function;
        empty string if called from an interactive environment (e.g., Jupyter notebook).
    - `{timestamp}`: The current date and time in the format `YYYYMMDD_HHMMSS`.

    Path-unsafe characters in the placeholder values are replaced with underscores, and for convenience,
    several additional variants of each placeholder are supported:

    - `{key-}`, `{key_}`: same as `{key}`, but with additional hyphen/underscore if the value is non-empty,
    - `{-key}`, `{_key}`: same as `{key}`, but with leading hyphen/underscore if the value is non-empty.

    Finally, both slashes and backslashes are replaced with the current OS path separator.

    Parameters:
      logdir_template: The log directory template with placeholders.
      **kwargs: The keyword arguments to fill the template.

    Returns:
      The formatted log directory path.

    Example:
      ```python
      parser = argparse.ArgumentParser()
      ...
      args = parser.parse_args()

      logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))
      ```
    """
    # Create {config} placeholder.
    items = [(re.sub("(.)[^-_]*[-_]?", r"\1", str(k)), str(v)) for k, v in sorted(kwargs.items())]
    if sum(len(k) + 1 + min(len(v), 5) + 1 for k, v in items) - 1 > 200:
        raise ValueError("Signature is too long to fit even with maximum truncation.")

    limit = max(len(v) for k, v in items)
    while sum(len(k) + 1 + min(len(v), limit) + 1 for k, v in items) - 1 > 200:  # guaranteed False when limit == 5
        limit -= 1
    items = [(k, v if len(v) <= limit else v[:limit // 2 - 1] + "..." + v[-limit // 2 + 2:]) for k, v in items]
    kwargs["config"] = ",".join(f"{k}={v}" for k, v in items)

    # Create {file} placeholder.
    current_frame = inspect.currentframe()
    if current_frame and current_frame.f_back:
        kwargs["file"] = os.path.splitext(os.path.basename(current_frame.f_back.f_globals.get("__file__")))[0]
    else:
        kwargs["file"] = ""

    # Create {timestamp} placeholder.
    kwargs["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize placeholder values and create variants.
    for key in list(kwargs.keys()):
        value = sanitize_path(str(kwargs[key]))
        kwargs[key] = value
        for separator in ["-", "_"]:
            kwargs[f"{key}{separator}"] = value and f"{value}{separator}"
            kwargs[f"{separator}{key}"] = value and f"{separator}{value}"

    return fill_and_standardize_path(logdir_template, **kwargs)
