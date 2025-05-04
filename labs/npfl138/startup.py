# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import random

import numpy as np
import torch


def startup(seed: int | None = None, threads: int | None = None, forkserver_instead_of_fork: bool = False) -> None:
    """Initialize the environment.

    - Allow using TF32 for matrix multiplication.
    - Set the random seed if given.
    - Set the number of threads if given.
    - Use `forkserver` instead of `fork` if requested.

    Parameters:
      seed: If not `None`, set the Python, Numpy, and PyTorch random seeds to this value.
      threads: If not `None` of 0, set the number of threads to this value.
        Otherwise, use as many threads as cores.
      forkserver_instead_of_fork: If `True`, use `forkserver` instead of `fork` as the
        default multiprocessing method. This will be the default in Python 3.14.
    """

    # Allow TF32 when available.
    torch.backends.cuda.matmul.allow_tf32 = True

    # Set random seed if not None.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Set number of threads if > 0; otherwise, use as many threads as cores.
    if threads is not None and threads > 0:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)

    # If instructed, use `forkserver` instead of `fork` (which will be the default in Python 3.14).
    if "fork" in torch.multiprocessing.get_all_start_methods():
        if os.environ.get("FORCE_FORK_METHOD") == "1":
            if torch.multiprocessing.get_start_method(allow_none=True) != "fork":
                torch.multiprocessing.set_start_method("fork")
        elif forkserver_instead_of_fork or os.environ.get("FORCE_FORKSERVER_METHOD") == "1":
            if torch.multiprocessing.get_start_method(allow_none=True) != "forkserver":
                torch.multiprocessing.set_start_method("forkserver")
