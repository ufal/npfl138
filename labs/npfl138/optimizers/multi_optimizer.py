# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections.abc import Iterable
from typing import Any

import torch


class MultiOptimizer(torch.optim.Optimizer):
    """A class managing multiple PyTorch optimizers, deferring all function calls to them.

    Warning:
        The implementation is quite hacky (MultiOptimizer does not call the parent constructor
        of the [torch.optim.Optimizer][] and provides only a subset of the functionality),
        but it seems to work well enough for [npfl138.optimizers.LazyAdam][] to work.

    Info:
        The current limitations of the MultiOptimizer are:

        - it does not provide `defaults` and `state` properties;
        - it does not support passing a `closure` to the `step()` method;
        - it does not support hooks.
    """
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        """Initializes the MultiOptimizer with a list of optimizers.

        Raises:
          ValueError: If the provided list of optimizers is empty,
            or if any parameter appears in more than one optimizer.
        """
        if not optimizers:
            raise ValueError("At least one optimizer must be provided")
        self.optimizers = list(optimizers)

        for i in range(len(self.optimizers)):
            for param_group_i in self.optimizers[i].param_groups:
                for j in range(i + 1, len(self.optimizers)):
                    for param_group_j in self.optimizers[j].param_groups:
                        overlap = set(param_group_i["params"]) & set(param_group_j["params"])
                        if overlap:
                            raise ValueError(f"Parameters appear in more than one optimizer: {overlap}")

    optimizers: list[torch.optim.Optimizer]
    """A list of optimizers managed by this MultiOptimizer instance."""

    @property
    def param_groups(self) -> list[dict]:
        """Returns a list of all parameter groups from all managed optimizers."""
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    @property
    def defaults(self) -> None:
        raise RuntimeError("MultiOptimizer does not have defaults")

    @property
    def state(self) -> None:
        raise RuntimeError("MultiOptimizer does not have state")

    def step(self, closure: None = None) -> None:
        """Performs a single optimization step for all managed optimizers.

        While the `closure` argument is accepted to follow [torch.optim.Optimizer.step][],
        it must be `None` because the MultiOptimizer cannot pass it to multiple optimizers.

        Parameters:
          closure: Must be `None`, while it might be `Callable[[], float]` in a
            [torch.optim.Optimizer.step][] method.
        """
        assert closure is None, "MultiOptimizer.step does not support a closure"
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Sets the gradients of all optimized parameters to zero for all managed optimizers.

        Parameters:
          set_to_none: See [torch.optim.Optimizer.zero_grad][].
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dictionary containing the state of all managed optimizers."""
        state = {}
        for i, optimizer in enumerate(self.optimizers):
            for key, value in optimizer.state_dict().items():
                state[f"optimizer_{i}_{key}"] = value
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dictionary in all managed optimizers."""
        for i, optimizer in enumerate(self.optimizers):
            optimizer_state_dict = {}
            for key in state_dict:
                if key.startswith(f"optimizer_{i}_"):
                    optimizer_state_dict[key.removeprefix(f"optimizer_{i}_")] = state_dict[key]
            optimizer.load_state_dict(optimizer_state_dict)
