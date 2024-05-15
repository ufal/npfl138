import os
import sys
from typing import Any, Callable, Iterator, TypedDict
import urllib.request

import numpy as np
import torch


class MNIST:
    H: int = 28
    W: int = 28
    C: int = 1
    LABELS: int = 10

    Element = TypedDict("Element", {"image": np.ndarray, "label": np.ndarray})
    Elements = TypedDict("Elements", {"images": np.ndarray, "labels": np.ndarray})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "MNIST.Elements", shuffle_batches: bool, seed: int = 42) -> None:
            self._data = data
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self) -> "MNIST.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "MNIST.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

        def transform(self, transform: Callable[["MNIST.Element"], Any]) -> "MNIST.TransformedDataset":
            return MNIST.TransformedDataset(self, transform)

        def batches(self, size: int | None = None) -> Iterator["MNIST.Element"]:
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "MNIST.TransformedDataset":
            return MNIST.TransformedDataset(self, transform)

    def __init__(self, dataset: str = "mnist", size: dict[str, int] = {}) -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: mnist[key][:size.get(dataset, None)]
                    for key in mnist if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))

    train: Dataset
    dev: Dataset
    test: Dataset
