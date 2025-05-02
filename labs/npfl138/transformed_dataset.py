# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Callable

import torch


class TransformedDataset(torch.utils.data.Dataset):
    """A dataset capable of applying transformations to its items and batches.

    """
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        """Create a new transformed dataset using the provided dataset.

        Parameters:
          dataset: The source dataset implementing `__len__` and `__getitem__`.
        """
        self._dataset = dataset

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the specified index."""
        item = self._dataset[index]
        if self.transform is not None:
            return self.transform(*item) if isinstance(item, tuple) else self.transform(item)
        return item

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        """Return the source dataset."""
        return self._dataset

    transform: Callable | None = None
    """If given, `transform` is called on each item before returning it.

    If the dataset item is a tuple, `transform` is called with the tuple unpacked.
    """

    collate: Callable | None = None
    """If given, `collate` is called on a list of items before returning them as a batch."""

    transform_batch: Callable | None = None
    """If given, `transform_batch` is called on a batch before returning it."""

    def collate_fn(self, batch: list[Any]) -> Any:
        """A function for a DataLoader to collate a batch of items using `collate` and/or `transform_batch`.

        This function is used as the `collate_fn` parameter of a DataLoader when `collate` or `transform_batch` is set.

        Parameters:
          batch: A list of items to collate and/or pass through `transform_batch`.
        """
        batch = self.collate(batch) if self.collate is not None else torch.utils.data.dataloader.default_collate(batch)
        if self.transform_batch is not None:
            batch = self.transform_batch(batch)
        return batch

    def dataloader(self, batch_size=1, *, shuffle=False, seed=None, num_workers=0, **kwargs) -> torch.utils.data.DataLoader:
        """Create a DataLoader for this dataset.

        This method is a convenience wrapper around [torch.utils.data.DataLoader][]
        setting up the required parameters. Most arguments are passed directly to the
        [torch.utils.data.DataLoader][], with a few exceptions:

        - When `seed` is given, it is used to construct the `generator` argument for the
          DataLoader using `torch.Generator().manual_seed(seed)`; the `generator` options
          must not be specified in `kwargs`.
        - When `shuffle` is `False` and no `generator` is given, `torch.Generator()` is passed
          as `generator`. Otherwise, the global random number generator would be used during
          every construction of an iterator, i.e. during every `iter(dataloader)` call.
        - When `num_workers` is greater than 0, `persistent_workers` is set to True.
        - When `collate` or `transform_batch` is set, the `self.collate_fn` is passed as the
          `collate_fn` parameter.
        """
        if seed is not None:
            # If a seed is given, create a generator with the seed and pass it to the DataLoader.
            if kwargs.get("generator", None) is not None:
                raise ValueError("When seed is given, generator must not be specified.")
            kwargs["generator"] = torch.Generator().manual_seed(seed)

        if not shuffle and kwargs.get("generator", None) is None:
            # If not shuffling and no generator is given, pass an explicit generator to the Dataloader.
            # Otherwise, the global random generator would generate a number on every iter(dataloader) call.
            kwargs["generator"] = torch.Generator()

        if num_workers > 0:
            # By default, set persistent_workers to True, but allow it to be overridden
            kwargs.setdefault("persistent_workers", True)

        if self.collate is not None or self.transform_batch is not None:
            if "collate_fn" in kwargs:
                raise ValueError("When collate or transform_batch is overridden, collate_fn must not be given.")
            kwargs["collate_fn"] = self.collate_fn

        # Create and return the DataLoader
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
