# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Callable

import torch


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        item = self._dataset[index]
        if self.transform is not None:
            return self.transform(*item) if isinstance(item, tuple) else self.transform(item)
        return item

    transform: Callable | None = None

    collate: Callable | None = None

    transform_batch: Callable | None = None

    def collate_fn(self, batch: list[Any]) -> Any:
        batch = self.collate(batch) if self.collate is not None else torch.utils.data.dataloader.default_collate(batch)
        if self.transform_batch is not None:
            batch = self.transform_batch(batch)
        return batch

    def dataloader(self, batch_size=1, *, shuffle=False, num_workers=0, **kwargs) -> torch.utils.data.DataLoader:
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
