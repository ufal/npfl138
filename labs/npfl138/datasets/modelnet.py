# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""ModelNet is a dataset containing 3D grids of voxelized objects.

The objects are available either as 20×20×20 or 32×32×32 voxel grids, and
are classified into 10 classes.

A visualization of a single object of every class is available both
for the [20×20×20](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/modelnet20.html)
and the [32×32×32](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/modelnet32.html)
resolutions.
"""
import os
import sys
from typing import Literal, Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch


class ModelNet:
    C: int = 1
    """The number of 3D grid channels."""
    D: int
    """The depth of the 3D grid, set in the constructor to 20 or 32."""
    H: int
    """The height of the 3D grid, set in the constructor to 20 or 32."""
    W: int
    """The width of the 3D grid, set in the constructor to 20 or 32."""
    LABELS: int = 10
    """The number of object classes."""
    LABEL_NAMES: list[str] = [
        "bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet",
    ]
    """The names of the object classes."""

    Element = TypedDict("Element", {"grid": np.ndarray, "label": np.ndarray})
    """The type of a single dataset element."""
    Elements = TypedDict("Elements", {"grids": np.ndarray, "labels": np.ndarray})
    """The type of the whole dataset."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/modelnet{}.npz"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "ModelNet.Elements") -> None:
            self._data = {key: torch.as_tensor(value) for key, value in data.items()}
            self._data["grids"] = self._data["grids"].movedim(-1, 1)

        @property
        def data(self) -> "ModelNet.Elements":
            """Return the whole dataset as a `ModelNet.Elements` object."""
            return self._data

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return len(self._data["grids"])

        def __getitem__(self, index: int) -> "ModelNet.Element":
            """Return the `index`-th element of the dataset."""
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

    def __init__(self, resolution: Literal[20, 32]) -> None:
        """Load the ModelNet dataset, downloading it if necessary.

        Parameters:
          resolution: The resolution of the dataset to load.
        """
        assert resolution in [20, 32], "Only 20 or 32 resolution is supported"

        self.D = self.H = self.W = resolution
        url = self.URL.format(resolution)

        path = os.path.basename(url)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(url, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        modelnet = np.load(path)
        for dataset, _size in [("train", 3_718), ("dev", 273), ("test", 908)]:
            data = dict((key[len(dataset) + 1:], modelnet[key]) for key in modelnet if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[int]) -> float:
        """Evaluate the `predictions` against the gold dataset.

        Returns:
          accuracy: The accuracy of the predictions in percentages.
        """
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          accuracy: The accuracy of the predictions in percentages.
        """
        predictions = [int(line) for line in predictions_file]
        return ModelNet.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--dim", default=20, type=int, help="ModelNet dimensionality to use")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = ModelNet.evaluate_file(getattr(ModelNet(args.dim), args.dataset), predictions_file)
        print("ModelNet accuracy: {:.2f}%".format(accuracy))
