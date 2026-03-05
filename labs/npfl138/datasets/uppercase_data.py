# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Loads the Uppercase data.

- The [UppercaseData][npfl138.datasets.uppercase_data.UppercaseData] consists of three Datasets
    - [`train`][npfl138.datasets.uppercase_data.UppercaseData.train]
    - [`dev`][npfl138.datasets.uppercase_data.UppercaseData.dev]
    - [`test`][npfl138.datasets.uppercase_data.UppercaseData.test]
- When loading, you need to specify `window` and `alphabet_size`. If
  `alphabet_size` is nonzero, it specifies the maximum number of alphabet
  characters, in which case that many most frequent characters will be used,
  and all other will be remapped to "&lt;unk>".
- Features are generated using a sliding window of a given size,
  i.e., for a character, we include left `window` characters, the character
  itself, and right `window` characters; `2 * window + 1` in total.
- Each dataset (train/dev/test) has the following members:
    - [`__len__`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.__len__]:
        the length of the text;
    - [`text`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.text]:
        the original text (of course lowercased in case of the test set);
    - [`alphabet`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.alphabet]:
        an alphabet used by [`windows`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.windows];
    - [`windows`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.windows]:
        a Pytorch Tensor with shape `[size, 2 * window + 1]` containing
        windows with indices of input lowercased characters;
    - [`labels`][npfl138.datasets.uppercase_data.UppercaseData.Dataset.labels]:
        a PyTorch Tensor with shape `[size]` containing 0/1 indicating whether
        the character of the corresponding window is lowercase/uppercase.
"""
import os
import sys
from typing import TextIO
import urllib.request
import zipfile

import numpy as np
import torch


class UppercaseData:
    LABELS: int = 2

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets/uppercase_data.zip"

    class Dataset:
        def __init__(self, data: str, window: int, alphabet: int | list[str]) -> None:
            self._window = window
            self._text = data
            self._size = len(data)

            # Create alphabet_map
            alphabet_map = {"<pad>": 0, "<unk>": 1}
            if not isinstance(alphabet, int):
                # Check that <pad> and <unk> are present at the beginning
                if alphabet[:2] == ["<pad>", "<unk>"]:
                    alphabet = alphabet[2:]
                else:
                    print("UppercaseData warning: The alphabet should start with <pad> and <unk>, prepending them.")

                for index, letter in enumerate(alphabet, len(alphabet_map)):
                    if letter in alphabet_map:
                        raise ValueError(f"UppercaseData: Duplicated character '{letter}' in the alphabet.")
                    alphabet_map[letter] = index
            else:
                # Find most frequent characters
                freqs = {}
                for char in self._text.lower():
                    freqs[char] = freqs.get(char, 0) + 1

                most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
                for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                    alphabet_map[char] = i
                    if alphabet and len(alphabet_map) >= alphabet:
                        break

            # Remap lowercased input characters using the alphabet_map and create labels
            lcletters = np.zeros(self._size + 2 * window, dtype=np.int64)
            labels = np.zeros(self._size, dtype=np.int64)
            for i in range(self._size):
                char = self._text[i].lower()
                if char not in alphabet_map:
                    char = "<unk>"
                lcletters[i + window] = alphabet_map[char]
                labels[i] = self._text[i].isupper()

            self._windows = torch.from_numpy(lcletters).unfold(0, 2 * window + 1, 1)
            self._labels = torch.from_numpy(labels)

            # Compute alphabet
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return self._size

        @property
        def text(self) -> str:
            """The original text of the dataset."""
            return self._text

        @property
        def alphabet(self) -> list[str]:
            """An alphabet used by `windows`."""
            return self._alphabet

        @property
        def windows(self) -> torch.Tensor:
            """A Tensor with shape `[size, 2 * window + 1]` and dtype `torch.int64`
            containing windows with indices of input lowercased characters.
            """
            return self._windows

        @property
        def labels(self) -> torch.Tensor:
            """A Tensor with shape `[size]` and dtype `torch.int64` containing zeros and ones
            indicating whether the character of the corresponding window is lowercase or uppercase.
            """
            return self._labels

    def __init__(self, window: int, alphabet_size: int = 0) -> None:
        """Load the UppercaseData dataset, downloading it if necessary.

        Parameters:
          window: The size of the sliding window of left and right characters to use for generating features.
          alphabet_size: If nonzero, the maximum number of alphabet characters (the most frequent ones will
            be used, others are remapped go "<unk>"); if zero, all characters are used.
        """
        path = os.path.basename(self.URL)
        if not os.path.exists(path):
            print(f"Downloading dataset {path}...", file=sys.stderr)
            urllib.request.urlretrieve(self.URL, filename=f"{path}.tmp")
            os.rename(f"{path}.tmp", path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}_{dataset}.txt", "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset, self.Dataset(
                    data,
                    window,
                    alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                ))

    train: Dataset
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset

    Warning:
        The test dataset is lowercased.
    """

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: str) -> float:
        """Evaluate the `predictions` against the gold dataset.

        Returns:
          accuracy
        """
        gold = gold_dataset.text

        if len(predictions) < len(gold):
            raise RuntimeError(f"The predictions are shorter than gold data: {len(predictions)} vs {len(gold)}.")

        correct = 0
        for i in range(len(gold)):
            # Note that just the lower() condition is not enough, for example
            # u03c2 and u03c3 have both u03c2 as an uppercase character.
            if predictions[i].lower() != gold[i].lower() and predictions[i].upper() != gold[i].upper():
                raise RuntimeError("The predictions and gold data differ on position {}: {} vs {}.".format(
                    i, repr(predictions[i:i + 20].lower()), repr(gold[i:i + 20].lower())))

            correct += gold[i] == predictions[i]
        return correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          accuracy
        """
        predictions = predictions_file.read()
        return UppercaseData.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = UppercaseData.evaluate_file(getattr(UppercaseData(0), args.dataset), predictions_file)
        print(f"Uppercase accuracy: {100 * accuracy:.2f}%")
