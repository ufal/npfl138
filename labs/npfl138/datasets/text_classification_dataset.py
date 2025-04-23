# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The `TextClassificationDataset` class represents a text classification dataset.

- Loads a text classification dataset in a vertical format.
- The data consists of three datasets:
    - `train`
    - `dev`
    - `test`
- Each dataset is a [torch.utils.data.Dataset][] providing
    - `__len__`: number of sentences in the dataset
    - `__getitem__`: return the requested sentence as an `Element`
      instance, which is a dictionary with keys "document" and "label",
      each being a string
    - `data`: a dictionary of type `Elements`, with keys "documents" and "labels"
    - `label_vocab`, a [npfl138.Vocabulary][] instance with the label mapping
"""
import os
import sys
from typing import Any, BinaryIO, Sequence, TextIO, TypedDict
import urllib.request
import zipfile
Self = Any  # For compatibility with Python <3.11 that does not support Self

import torch

from ..vocabulary import Vocabulary


class TextClassificationDataset:
    Element = TypedDict("Element", {"document": str, "label": str})
    """The type of a single dataset element, i.e., a single document and its label."""
    Elements = TypedDict("Elements", {"documents": list[str], "labels": list[str]})
    """The type of the whole dataset, i.e., a corpus of documents."""

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_file: BinaryIO, train: Self | None = None) -> None:
            # Load the data
            self._data: "TextClassificationDataset.Elements" = {
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
            """Return the whole dataset as a `TextClassificationDataset.Elements` object."""
            return self._data

        @property
        def label_vocab(self) -> Vocabulary:
            """The label vocabulary of the dataset."""
            return self._label_vocab

        def __len__(self) -> int:
            """Return the number of documents in the dataset."""
            return len(self._data["labels"])

        def __getitem__(self, index: int) -> "TextClassificationDataset.Element":
            """Return the `index`-th element of the dataset as a dictionary."""
            return {key.removesuffix("s"): value[index] for key, value in self._data.items()}

    def __init__(self, name: str) -> None:
        """Create the dataset from the given filename, downloading it if necessary."""
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
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[str]) -> float:
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
