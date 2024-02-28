import os
import sys
from typing import Dict, Iterator, Optional
import urllib.request

import numpy as np


class MNIST:
    H: int = 28
    W: int = 28
    C: int = 1
    LABELS: int = 10

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Datasplit:
        def __init__(self, data: Dict[str, np.ndarray], shuffle_batches: bool, seed: int = 42) -> None:
            self._data = data
            self._size = len(self._data["images"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self) -> Dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        def batches(self, size: Optional[int] = None) -> Iterator[Dict[str, np.ndarray]]:
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    def __init__(self, dataset: str = "mnist", size: Dict[str, int] = {}) -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: mnist[key][:size.get(dataset, None)]
                    for key in mnist if key.startswith(dataset)}
            setattr(self, dataset, self.Datasplit(data, shuffle_batches=dataset == "train"))

    train: Datasplit
    dev: Datasplit
    test: Datasplit
