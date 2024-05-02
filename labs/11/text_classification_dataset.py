import os
import sys
from typing import Any, BinaryIO, Callable, Iterable, Self, Sequence, TextIO, TypedDict
import urllib.request
import zipfile

import torch


# A class for managing mapping between strings and indices.
# It provides:
# - `__len__`: number of strings in the vocabulary
# - `__iter__`: iterator over strings in the vocabulary
# - `string(index: int) -> str`: string for a given index to the vocabulary
# - `strings(indices: Sequence[int]) -> list[str]`: list of strings for given indices
# - `index(string: str) -> int`: index of a given string in the vocabulary
# - `indices(strings: Sequence[str]) -> list[int]`: list of indices for given strings
class Vocabulary:
    PAD: int = 0
    UNK: int = 1

    def __init__(self, strings: Sequence[str]) -> None:
        self._strings = ["[PAD]", "[UNK]"]
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}

    def __len__(self) -> int:
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        return iter(self._strings)

    def string(self, index: int) -> str:
        return self._strings[index]

    def strings(self, indices: Sequence[int]) -> list[str]:
        return [self._strings[index] for index in indices]

    def index(self, string: str) -> int:
        return self._string_map.get(string, Vocabulary.UNK)

    def indices(self, strings: Sequence[str]) -> list[int]:
        return [self._string_map.get(string, Vocabulary.UNK) for string in strings]


# Loads a text classification dataset in a vertical format.
# - The data consists of three datasets
#  - `train`
#  - `dev`
#  - `test`
# - Each dataset is a `torch.utils.data.Dataset` providing
#   - `__len__`: number of sentences in the dataset
#   - `__getitem__`: return the requested sentence as an `Element`
#     instance, which is a dictionary with keys "document" and "label",
#     each being a string
#   - `data`: a dictionary of type `Elements`, with keys "documents" and "labels"
#   - `label_vocab`, a `Vocabulary` instance with the label mapping
class TextClassificationDataset:
    Element = TypedDict("Element", {"document": str, "label": str})
    Elements = TypedDict("Elements", {"documents": list[str], "labels": list[str]})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_file: BinaryIO, train: Self | None = None) -> None:
            # Load the data
            self._data: Elements = {
                "documents": [],
                "labels": [],
            }
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                label, document = line.split("\t", maxsplit=1)

                self._data["documents"].append(document)
                self._data["labels"].append(label)

            # Create or copy the label mapping
            if train:
                self._label_vocab = train._label_vocab
            else:
                self._label_vocab = Vocabulary(sorted(set(self._data["labels"])))

        @property
        def data(self) -> "TextClassificationDataset.Elements":
            return self._data

        @property
        def label_vocab(self) -> Vocabulary:
            return self._label_vocab

        def __len__(self) -> int:
            return len(self._data["labels"])

        def __getitem__(self, index: int) -> "TextClassificationDataset.Element":
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

        def transform(
            self, transform: Callable[["TextClassificationDataset.Element"], Any]
        ) -> "TextClassificationDataset.TransformedDataset":
            return TextClassificationDataset.TransformedDataset(self, transform)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "TextClassificationDataset.TransformedDataset":
            return TextClassificationDataset.TransformedDataset(self, transform)

    def __init__(self, name: str) -> None:
        """Create the dataset from the given filename."""
        path = "{}.zip".format(name)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file, train=getattr(self, "train", None)))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[str]) -> float:
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [line.rstrip("\r\n") for line in predictions_file]
        return TextClassificationDataset.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="czech_facebook", type=str, help="Text classification corpus")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = TextClassificationDataset.evaluate_file(
                getattr(TextClassificationDataset(args.corpus), args.dataset), predictions_file)
        print("Text classification accuracy: {:.2f}%".format(accuracy))
