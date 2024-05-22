import os
import sys
from typing import Any, Callable, TypedDict
import urllib.request

import numpy as np
import torch

class Omniglot:
    H: int = 28
    W: int = 28
    C: int = 1

    Element = TypedDict("Element", {
        "image": np.ndarray, "label": np.ndarray, "alphabet": np.ndarray, "alphabet_char_id": np.ndarray})
    Elements = TypedDict("Elements", {
        "images": np.ndarray, "labels": np.ndarray, "alphabets": np.ndarray, "alphabet_char_ids": np.ndarray})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "Omniglot.Elements") -> None:
            self._data = data
            self._size = len(self._data["images"])

        @property
        def data(self) -> "Omniglot.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "Omniglot.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

        def transform(self, transform: Callable[["Omniglot.Element"], Any]) -> "Omniglot.TransformedDataset":
            return Omniglot.TransformedDataset(self, transform)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "Omniglot.TransformedDataset":
            return Omniglot.TransformedDataset(self, transform)

    def __init__(self, dataset: str = "omniglot") -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        omniglot = np.load(path)
        for dataset in ["train", "test"]:
            data = {key[len(dataset) + 1:]: omniglot[key] for key in omniglot if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    test: Dataset
