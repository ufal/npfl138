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
    """The number of image channels."""
    H: int = 28
    """The image height."""
    W: int = 28
    """The image width."""
    LABELS: int = 10
    """The number of labels."""

    Element = TypedDict("Element", {"image": torch.Tensor, "label": torch.Tensor})
    """The type of a single dataset element."""
    Elements = TypedDict("Elements", {"images": torch.Tensor, "labels": torch.Tensor})
    """The type of the whole dataset."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "MNIST.Elements") -> None:
            self._data = {key: torch.as_tensor(value) for key, value in data.items()}
            self._data["images"] = self._data["images"].view(-1, MNIST.C, MNIST.H, MNIST.W)

        @property
        def data(self) -> "MNIST.Elements":
            """Return the whole dataset as a `MNIST.Elements` object."""
            return self._data

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "MNIST.Element":
            """Return the `index`-th element of the dataset."""
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
        """Load the MNIST dataset, downloading it if necessary.

        Parameters:
          dataset: The name of the dataset, typically `mnist`.
          sizes: An optional dictionary overriding the sizes of the `train`, `dev`, and `test` splits.
        """
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
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""
