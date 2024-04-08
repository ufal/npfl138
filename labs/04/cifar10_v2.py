import os
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch


class CIFAR10:
    H: int = 32
    W: int = 32
    C: int = 3
    LABELS: list[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    Element = TypedDict("Element", {"image": np.ndarray, "label": np.ndarray})
    Elements = TypedDict("Elements", {"images": np.ndarray, "labels": np.ndarray})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/cifar10_competition.npz"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data: "CIFAR10.Elements") -> None:
            self._data = data
            self._data["labels"] = self._data["labels"].ravel()

        @property
        def data(self) -> "CIFAR10.Elements":
            return self._data

        def __len__(self) -> int:
            return len(self._data["images"])

        def __getitem__(self, index: int) -> "CIFAR10.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

        def transform(self, transform: Callable[["CIFAR10.Element"], Any]) -> "CIFAR10.TransformedDataset":
            return CIFAR10.TransformedDataset(self, transform)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "CIFAR10.TransformedDataset":
            return CIFAR10.TransformedDataset(self, transform)

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
