# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import random

import numpy as np
import torch


def startup(seed: int | None = None,
            threads: int | None = None,
            forkserver_instead_of_fork: bool = False,
            recodex: bool = False) -> None:
    """Initialize the environment.

    - Allow using TF32 for matrix multiplication.
    - Set the random seed if given.
    - Set the number of threads if given.
    - Use `forkserver` instead of `fork` if requested.
    - Optionally run in ReCodEx mode for better replicability. In this mode,
        - Layer initialization does not depend on the global random seed generator
          (it is deterministic and depends only on the parameter shape).
        - Every dataloader uses its own random generator.
        - However, the deterministic layer initialization decreases performance
          of trained models, so it should be used only for running tests.

    Parameters:
      seed: If not `None`, set the Python, Numpy, and PyTorch random seeds to this value.
      threads: If not `None` of 0, set the number of threads to this value.
        Otherwise, use as many threads as cores.
      forkserver_instead_of_fork: If `True`, use `forkserver` instead of `fork` as the
        default multiprocessing method. This will be the default in Python 3.14.
      recodex: If `True`, run in ReCodEx mode for better replicability of tests.
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

    # If ReCodEx mode is requested, apply various overrides for better replicability.
    if recodex:
        # Do not use accelerators.
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: False
        torch.xpu.is_available = lambda: False

        # Make initializers deterministic.
        def bind_generator(init):
            return lambda *args, **kwargs: init(*args, **kwargs | {"generator": torch.Generator().manual_seed(seed)})
        for name in ["uniform_", "normal_", "trunc_normal_", "xavier_uniform_", "xavier_normal_",
                     "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_"]:
            setattr(torch.nn.init, name, bind_generator(getattr(torch.nn.init, name)))

        # Override the generator of every DataLoader to a fresh default generator.
        original_dataloader_init = torch.utils.data.DataLoader.__init__
        torch.utils.data.DataLoader.__init__ = lambda self, dataset, *args, **kwargs: original_dataloader_init(
            self, dataset, *args, **kwargs | {"generator": torch.Generator().manual_seed(seed)})
