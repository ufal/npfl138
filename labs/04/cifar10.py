import os
import sys
from typing import Any, Callable, Sequence, TextIO
import urllib.request

import numpy as np
import torch


class CIFAR10:
    H: int = 32
    W: int = 32
    C: int = 3
    LABELS: list[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/cifar10_competition.npz"

    class Dataset:
        def __init__(self, data: dict[str, np.ndarray]) -> None:
            self._data = data
            self._data["labels"] = self._data["labels"].ravel()
            self._size = len(self._data["images"])

        @property
        def data(self) -> dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        def dataset(self, transform: Callable[[dict[str, np.ndarray]], Any] | None = None) -> torch.utils.data.Dataset:
            return CIFAR10.TorchDataset(self, transform)

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: "CIFAR10.Dataset",
                     transform: Callable[[dict[str, np.ndarray]], Any] | None) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return self._dataset.size

        def __getitem__(self, index: int) -> Any:
            item = {key: value[index] for key, value in self._dataset.data.items()}
            if self._transform is not None:
                item = self._transform(item)
            return item

    def __init__(self, size: dict[str, int] = {}) -> None:
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading CIFAR-10 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        cifar = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = {key[len(dataset) + 1:]: cifar[key][:size.get(dataset, None)]
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

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
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
