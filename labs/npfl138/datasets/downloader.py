# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os
import sys

import torch


def download_url_to_file(url_dir: str, filename: str) -> str:
    """Download the file from a given URL directory and save it as a given filename.

    Parameters:
      url: The URL of a directory where to download the file from.
      filename: The name of the file to save the downloaded content to.

    Returns:
      The path to the file the content was saved to.
    """
    cache_dir = os.environ.get("NPFL_CACHE")

    path = os.path.join(cache_dir, filename) if cache_dir is not None else filename

    if not os.path.exists(path):
        print(f"Downloading {filename}...", file=sys.stderr)
        torch.hub.download_url_to_file(url=f"{url_dir}/{filename}", dst=path, progress=True)

    return path
