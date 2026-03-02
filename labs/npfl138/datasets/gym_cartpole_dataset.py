# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The `GymCartpoleDataset` is a toy dataset with 100 samples from the CartPole environment.

The dataset consists of just a single subset:

- `train`: 100 examples, each a `(observation, label)` pair, where
    - `inputs` is a vector with [npfl138.datasets.gym_cartpole_dataset.GymCartpoleDataset.FEATURES][]
      floating point values,
    - `label` is a gold 0/1 class index.
"""
import collections
import os
import sys
import urllib.request

import numpy as np
import torch


class GymCartpoleDataset:
    FEATURES: int = 4
    """Number of features of every example in the dataset."""

    Element = collections.namedtuple("Element", ["observation", "label"])
    """The type of a single dataset element."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, observations: np.ndarray, labels: np.ndarray) -> None:
            self._observations = torch.as_tensor(observations)
            self._labels = torch.as_tensor(labels)

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return len(self._observations)

        def __getitem__(self, index: int) -> "GymCartpoleDataset.Element":
            """Return the `index`-th element of the dataset."""
            return GymCartpoleDataset.Element(self._observations[index], self._labels[index])

        @property
        def observations(self) -> torch.Tensor:
            """All dataset observations as a single tensor."""
            return self._observations

        @property
        def labels(self) -> torch.Tensor:
            """All dataset labels as a single tensor."""
            return self._labels

    def __init__(self, dataset: str = "gym_cartpole_data.txt") -> None:
        """Load the GymCartpoleDataset, downloading it if necessary.

        Parameters:
          dataset: The file name of the dataset to load.
        """
        if not os.path.exists(dataset):
            print(f"Downloading {dataset}...", file=sys.stderr)
            urllib.request.urlretrieve(f"{self.URL}/{dataset}", filename=f"{dataset}.tmp")
            os.rename(f"{dataset}.tmp", dataset)

        data = np.loadtxt(dataset)
        self.train = self.Dataset(observations=data[:, :-1].astype(np.float32), labels=data[:, -1].astype(np.int64))

    train: Dataset
    """The training dataset."""
