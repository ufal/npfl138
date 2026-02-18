# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Callable
import os
from typing import Any

import torch

from .utils import tuple_list


class TransformedDataset(torch.utils.data.Dataset):
    """A dataset capable of applying transformations to its items and batches.

    Assuming the `TransformedDataset` is used within a [DataLoader][torch.utils.data.DataLoader],
    batches are produced as follows:

    - First, every element is retrieved from the source dataset using `__getitem__` (or a list of
      all batch elements at the same time using `__getitems__`).
    - Then, if [transform][npfl138.TransformedDataset.transform] is defined, it is applied to each individual item.
    - Next, if [collate][npfl138.TransformedDataset.collate] is defined, it is applied to the list of items to form
      a batch.
    - Finally, if [transform_batch][npfl138.TransformedDataset.transform_batch] is defined, it is applied to the batch.

    Warning:
      Given how PyTorch [torch.utils.data.DataLoader][] works, when specifying
      [collate][npfl138.TransformedDataset.collate] or [transform_batch][npfl138.TransformedDataset.transform_batch],
      the `collate_fn` of the DataLoader **must** be set to `self.collate_fn`.

      This is automatically done when using the [dataloader][npfl138.TransformedDataset.dataloader]
      method of this class to create the [DataLoader][torch.utils.data.DataLoader]. However,
      if you create a [DataLoader][torch.utils.data.DataLoader] manually, you **must** pass
      `collate_fn=transformed_dataset.collate_fn` or otherwise [collate][npfl138.TransformedDataset.collate]
      and [transform_batch][npfl138.TransformedDataset.transform_batch] will be ignored.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, dataset_limit: int | None = None) -> None:
        """Create a new transformed dataset using the provided dataset with an optional limit.

        Parameters:
          dataset: The source dataset implementing `__len__` and `__getitem__`.
          dataset_limit: If given, limits the length of the dataset to this value.

        **Environment variables:** The following environment variable can be used
        to override the method parameters:

        - `NPFL_DATASET_LIMIT`: If set to a positive integer, overrides the `dataset_limit` parameter.
        """
        self._dataset = dataset
        self._dataset_has_getitems = hasattr(self._dataset, "__getitems__")

        if os.environ.get("NPFL_DATASET_LIMIT", "").isdecimal():
            dataset_limit = int(os.environ["NPFL_DATASET_LIMIT"])
        if dataset_limit is not None and dataset_limit < len(self._dataset):
            self._dataset = torch.utils.data.Subset(self._dataset, list(range(dataset_limit)))

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the specified index."""
        item = self._dataset[index]
        if self.transform is not None:
            return self.transform(*item) if isinstance(item, tuple_list) else self.transform(item)
        return item

    def __getitems__(self, indices: list[int]) -> Any:
        """Return a batch of items at the specified indices."""
        if self._dataset_has_getitems:
            batch = self._dataset.__getitems__(indices)
        else:
            batch = [self._dataset[i] for i in indices]
        if self.transform is not None:
            batch = [self.transform(*item) if isinstance(item, tuple_list) else self.transform(item) for item in batch]
        return batch

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        """Return the source dataset."""
        return self._dataset

    transform: Callable | None = None
    """If given, `transform` is called on each item before returning it.

    If the dataset item is a tuple or a list, `transform` is called with it unpacked.
    """

    collate: Callable | None = None
    """If given, `collate` is called on a list of items before returning them as a batch."""

    transform_batch: Callable | None = None
    """If given, `transform_batch` is called on a batch before returning it.

    If the batch is a tuple or a list, `transform_batch` is called with it unpacked.
    """

    def collate_fn(self, batch: list[Any]) -> Any:
        """A function for a DataLoader to collate a batch of items using `collate` and/or `transform_batch`.

        This function is used as the `collate_fn` parameter of a DataLoader when `collate` or `transform_batch` is set.

        Parameters:
          batch: A list of items to collate and/or pass through `transform_batch`.
        """
        batch = self.collate(batch) if self.collate is not None else torch.utils.data.dataloader.default_collate(batch)
        if self.transform_batch is not None:
            batch = self.transform_batch(*batch) if isinstance(batch, tuple_list) else self.transform_batch(batch)
        return batch

    def dataloader(
        self, batch_size=1, *, shuffle=False, seed=None, num_workers=0, **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
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
