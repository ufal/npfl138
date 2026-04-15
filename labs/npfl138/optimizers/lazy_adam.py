# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Iterable

import torch

from .multi_optimizer import MultiOptimizer


class LazyAdam(MultiOptimizer):
    """A class implementing the LazyAdam optimizer.

    The optimizer applies [torch.optim.SparseAdam][] to parameters of all [torch.nn.Embedding][] and
    [torch.nn.EmbeddingBag][] layers with sparse gradients, and [torch.optim.Adam][] to all other parameters.
    By default, all embedding layers in the module are set to have sparse gradients first.

    Warning:
        The implementation of [MultiOptimizer][npfl138.optimizers.MultiOptimizer] is quite hacky (it does not call
        the parent constructor of the [torch.optim.Optimizer][] and provides only a subset of the functionality),
        but it seems to work well enough for `LazyAdam` to work.

    Info:
        The current limitations of the [MultiOptimizer][npfl138.optimizers.MultiOptimizer] and thus `LazyAdam` are:

        - it does not provide `defaults` and `state` properties;
        - it does not support passing a `closure` to the `step()` method;
        - it does not support hooks.
    """
    def __init__(
        self,
        module: torch.nn.Module,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,
        *,
        make_embeddings_sparse: bool = True,
        adam_param_groups: Iterable | None = None,
    ) -> None:
        """Initializes the LazyAdam optimizer.

        Parameters:
          module: The module containing the embedding and non-embedding layers to be optimized.
          lr: The learning rate for both optimizers. Default is `0.001`.
          betas: The beta coefficients for both optimizers. Default is `(0.9, 0.999)`.
          eps: The epsilon value for both optimizers. Beware that the default value `1e-7` is
            different from `eps=1e-8` in [torch.optim.Adam][] and [torch.optim.SparseAdam][].
          make_embeddings_sparse: If `True` (default), sets the `sparse` attribute of all
            [torch.nn.Embedding][] and [torch.nn.EmbeddingBag][] layers to `True`; otherwise `LazyAdam` will
            consider only those layers that already have `sparse=True`.
          adam_param_groups: An optional iterable of parameters to optimize using [torch.optim.Adam][].
            If `None` (default), all `module.parameters()` that are not part of embedding layers with
            sparse gradients will be optimized using [torch.optim.Adam][].
        """
        sparse_params = []

        def collect_sparse_params(m: torch.nn.Module) -> None:
            if isinstance(m, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                if make_embeddings_sparse:
                    m.sparse = True
                if m.sparse:
                    nonlocal sparse_params
                    sparse_params.extend(m.parameters())
        module.apply(collect_sparse_params)
        assert sparse_params, "No embedding layers with sparse gradients found in the module."

        if adam_param_groups is None:
            adam_param_groups = []
            sparse_param_set = set(sparse_params)
            for param in module.parameters():
                if param not in sparse_param_set:
                    adam_param_groups.append(param)

        self.adam = torch.optim.Adam(adam_param_groups, lr=lr, betas=betas, eps=eps)
        self.sparse_adam = torch.optim.SparseAdam(sparse_params, lr=lr, betas=betas, eps=eps)

        super().__init__([self.adam, self.sparse_adam])

    adam: torch.optim.Optimizer
    """The [torch.optim.Adam][] optimizer for non-embedding parameters."""

    sparse_adam: torch.optim.Optimizer
    """The [torch.optim.SparseAdam][] optimizer for embedding parameters."""
