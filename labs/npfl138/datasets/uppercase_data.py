# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Loads the Uppercase data.

- The `UppercaseData` consists of three Datasets
    - `train`
    - `dev`
    - `test`
- When loading, you need to specify `window` and `alphabet_size`. If
  `alphabet_size` is nonzero, it specifies the maximum number of alphabet
  characters, in which case that many most frequent characters will be used,
  and all other will be remapped to "<unk>".
- Features are generated using a sliding window of given size,
  i.e., for a character, we include left `window` characters, the character
  itself and right `window` characters, `2 * window + 1` in total.
- Each dataset (train/dev/test) has the following members:
    - `size`: the length of the text
    - `text`: the original text (of course lowercased in case of the test set)
    - `alphabet`: an alphabet used by `windows`
    - `windows`: a torch Tensor with shape `[size, 2 * window + 1]` containing
      windows with indices of input lowercased characters
    - `labels`: a torch Tensor with shape `[size]` containing 0/1 for lower/uppercase
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

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/uppercase_data.zip"

    class Dataset:
        def __init__(self, data: str, window: int, alphabet: int | list[str], label_dtype: torch.dtype) -> None:
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
                        raise ValueError("UppercaseData: Duplicated character '{}' in the alphabet.".format(letter))
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
            labels = np.zeros(self._size, dtype=np.float32)
            for i in range(self._size):
                char = self._text[i].lower()
                if char not in alphabet_map:
                    char = "<unk>"
                lcletters[i + window] = alphabet_map[char]
                labels[i] = self._text[i].isupper()

            self._windows = torch.from_numpy(lcletters).unfold(0, 2 * window + 1, 1)
            self._labels = torch.from_numpy(labels).to(dtype=label_dtype)

            # Compute alphabet
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

        @property
        def size(self) -> int:
            return self._size

        @property
        def text(self) -> str:
            return self._text

        @property
        def alphabet(self) -> list[str]:
            return self._alphabet

        @property
        def windows(self) -> torch.Tensor:
            return self._windows

        @property
        def labels(self) -> torch.Tensor:
            return self._labels

    def __init__(self, window: int, alphabet_size: int = 0, label_dtype: torch.dtype = torch.float32) -> None:
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset, self.Dataset(
                    data,
                    window,
                    alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                    label_dtype=label_dtype,
                ))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: str) -> float:
        gold = gold_dataset.text

        if len(predictions) < len(gold):
            raise RuntimeError("The predictions are shorter than gold data: {} vs {}.".format(
                len(predictions), len(gold)))

        correct = 0
        for i in range(len(gold)):
            # Note that just the lower() condition is not enough, for example
            # u03c2 and u03c3 have both u03c2 as an uppercase character.
            if predictions[i].lower() != gold[i].lower() and predictions[i].upper() != gold[i].upper():
                raise RuntimeError("The predictions and gold data differ on position {}: {} vs {}.".format(
                    i, repr(predictions[i:i + 20].lower()), repr(gold[i:i + 20].lower())))

            correct += gold[i] == predictions[i]
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = predictions_file.read()
        return UppercaseData.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = UppercaseData.evaluate_file(getattr(UppercaseData(0), args.dataset), predictions_file)
        print("Uppercase accuracy: {:.2f}%".format(accuracy))
