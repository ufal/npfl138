# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import sys
from typing import Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch


class CIFAR10:
    C: int = 3
    H: int = 32
    W: int = 32
    LABELS: int = 10
    LABEL_NAMES: list[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    Element = TypedDict("Element", {"image": torch.Tensor, "label": torch.Tensor})
    Elements = TypedDict("Elements", {"images": torch.Tensor, "labels": torch.Tensor})

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/cifar10_competition.npz"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "CIFAR10.Elements") -> None:
            self._data = {key: torch.as_tensor(value) for key, value in data.items()}
            self._data["images"] = self._data["images"].moveaxis(-1, 1)
            self._data["labels"] = self._data["labels"].view(-1)

        @property
        def data(self) -> "CIFAR10.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "CIFAR10.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

    def __init__(self, sizes: dict[str, int] = {}) -> None:
        path = os.path.basename(self.URL)
        if not os.path.exists(path):
            print("Downloading CIFAR-10 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self.URL, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        cifar = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: cifar[key][:sizes.get(dataset, None)]
                    for key in cifar if key.startswith(dataset)}
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[int]) -> float:
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(bool(gold[i] == predictions[i]) for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [int(line) for line in predictions_file]
        return CIFAR10.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = CIFAR10.evaluate_file(getattr(CIFAR10(), args.dataset), predictions_file)
        print("CIFAR10 accuracy: {:.2f}%".format(accuracy))
