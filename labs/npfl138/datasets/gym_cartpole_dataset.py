# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import collections
import os
import sys
import urllib.request

import numpy as np
import torch


class GymCartpoleDataset:
    """Toy dataset with 100 noisy samples from the CartPole environment."""
    FEATURES: int = 4

    Element = collections.namedtuple("Element", ["observation", "label"])

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, observations: np.ndarray, labels: np.ndarray) -> None:
            self._observations = torch.as_tensor(observations)
            self._labels = torch.as_tensor(labels)

        def __len__(self) -> int:
            return len(self._observations)

        def __getitem__(self, index: int) -> "GymCartpoleDataset.Element":
            return GymCartpoleDataset.Element(self._observations[index], self._labels[index])

        @property
        def observations(self) -> torch.Tensor:
            return self._observations

        @property
        def labels(self) -> torch.Tensor:
            return self._labels

    def __init__(self, dataset: str = "gym_cartpole_data.txt") -> None:
        if not os.path.exists(dataset):
            print("Downloading {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self.URL, dataset), filename="{}.tmp".format(dataset))
            os.rename("{}.tmp".format(dataset), dataset)

        data = np.loadtxt(dataset)
        self.train = self.Dataset(observations=data[:, :-1].astype(np.float32), labels=data[:, -1].astype(np.int64))

    train: Dataset
