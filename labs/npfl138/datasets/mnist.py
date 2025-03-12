# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import sys
from typing import Iterator, TypedDict
import urllib.request

import numpy as np
import torch


class MNIST:
    C: int = 1
    H: int = 28
    W: int = 28
    LABELS: int = 10

    Element = TypedDict("Element", {"image": torch.Tensor, "label": torch.Tensor})
    Elements = TypedDict("Elements", {"images": torch.Tensor, "labels": torch.Tensor})

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "MNIST.Elements") -> None:
            self._data = {key: torch.as_tensor(value) for key, value in data.items()}
            self._data["images"] = self._data["images"].view(-1, MNIST.C, MNIST.H, MNIST.W)

        @property
        def data(self) -> "MNIST.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "MNIST.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

        def batches(
            self, size: int, shuffle: bool = False, generator: torch.Generator | None = None,
        ) -> Iterator["MNIST.Element"]:
            permutation = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))

            while len(permutation):
                batch_size = min(size, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {key: value[batch_perm] for key, value in self._data.items()}
                yield batch

    def __init__(self, dataset: str = "mnist", sizes: dict[str, int] = {}) -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self.URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: mnist[key][:sizes.get(dataset, None)]
                    for key in mnist if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    dev: Dataset
    test: Dataset
