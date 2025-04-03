# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The `MorphoDataset` class loads a morphological dataset in a vertical format.

- The data consists of three datasets
    - `train`
    - `dev`
    - `test`
- Each dataset is a [torch.utils.data.Dataset][] providing
    - `__len__`: number of sentences in the dataset
    - `__getitem__`: return the requested sentence as an `Element`
      instance, which is a dictionary with keys "forms"/"lemmas"/"tags",
      each being a list of strings
    - `forms`, `lemmas`, `tags`: instances of type `Factor` containing
      the following fields:
        - `strings`: a Python list containing input sentences, each being
          a list of strings (forms/lemmas/tags)
        - `word_vocab`: a [npfl138.Vocabulary][] object capable of mapping words to
          indices. It is constructed on the train set and shared by the dev
          and test sets
        - `char_vocab`: a [npfl138.Vocabulary][] object capable of mapping characters
          to  indices. It is constructed on the train set and shared by the dev
          and test sets
    - `cle_batch`: a method for creating inputs for character-level embeddings.
      It takes a list of sentences, each being a list of string forms, and produces
      a tuple of two tensors:
        - `unique_forms` with shape `[num_unique_forms, max_form_length]` containing
          each unique form as a sequence of character ids
        - `forms_indices` with shape `[num_sentences, max_sentence_length]`
          containing for every form its index in `unique_forms`
"""
import os
import sys
from typing import BinaryIO, Self, Sequence, TextIO, TypedDict
import urllib.request
import zipfile

import torch

from ..vocabulary import Vocabulary


class MorphoDataset:
    PAD: int = 0
    """The index of the padding token in the vocabulary, always present."""
    UNK: int = 1
    """The index of the unknown token in the vocabulary, always present."""
    BOW: int = 2
    """A special beginning-of-word token, always present in character vocabularies."""
    EOW: int = 3
    """A special end-of-word token, always present in character vocabularies."""

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    Element = TypedDict("Element", {"forms": list[str], "lemmas": list[str], "tags": list[str]})
    """The type of a single dataset element, i.e., a single sentence."""

    class Factor:
        """A factor of the dataset, i.e., forms, lemmas or tags."""
        word_vocab: Vocabulary
        """The word vocabulary of this factor."""
        char_vocab: Vocabulary
        """The character vocabulary of this factor."""
        strings: list[list[str]]
        """The input sentences of this factor, each being a list of strings."""

        def __init__(self) -> None:
            self.strings = []

        def finalize(self, train: Self | None = None) -> None:
            # Create vocabularies
            if train:
                self.word_vocab = train.word_vocab
                self.char_vocab = train.char_vocab
            else:
                strings = sorted(set(string for sentence in self.strings for string in sentence))
                self.word_vocab = Vocabulary(strings)

                bow_eow = ["[BOW]", "[EOW]"]
                self.char_vocab = Vocabulary(bow_eow + sorted(set(char for string in strings for char in string)))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_file: BinaryIO, train: Self | None = None, max_sentences: int | None = None) -> None:
            # Create factors
            self._factors = (MorphoDataset.Factor(), MorphoDataset.Factor(), MorphoDataset.Factor())
            self._factors_tensors = None

            # Load the data
            self._size = 0
            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_sentence:
                        for factor in self._factors:
                            factor.strings.append([])
                        self._size += 1

                    columns = line.split("\t")
                    assert len(columns) == len(self._factors)
                    for column, factor in zip(columns, self._factors):
                        factor.strings[-1].append(column)

                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and self._size >= max_sentences:
                        break

            # Finalize the mappings
            for i, factor in enumerate(self._factors):
                factor.finalize(train._factors[i] if train else None)

        @property
        def forms(self) -> "MorphoDataset.Factor":
            """Factor containing the forms of the dataset."""
            return self._factors[0]

        @property
        def lemmas(self) -> "MorphoDataset.Factor":
            """Factor containing the lemmas of the dataset."""
            return self._factors[1]

        @property
        def tags(self) -> "MorphoDataset.Factor":
            """Factor containing the tags of the dataset."""
            return self._factors[2]

        def __len__(self) -> int:
            """Return the number of sentences in the dataset."""
            return self._size

        def __getitem__(self, index: int) -> "MorphoDataset.Element":
            """Return the `index`-th element of the dataset as a dictionary."""
            return {"forms": self.forms.strings[index],
                    "lemmas": self.lemmas.strings[index],
                    "tags": self.tags.strings[index]}

        def cle_batch(self, forms: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
            """Create a batch suitable for computation of character-level word embeddings.

            Parameters:
              forms: A batch of sentences, each being a list of string forms.

            Returns:
              unique_forms: A tensor with shape `[num_unique_forms, max_form_length]`
                containing each unique form as a sequence of character ids.
              forms_indices: A tensor with shape `[num_sentences, max_sentence_length]`
                containing for every form from the batch its index in `unique_forms`.
            """
            unique_strings = list(set(form for sentence in forms for form in sentence))
            unique_string_map = {form: index + 1 for index, form in enumerate(unique_strings)}
            unique_forms = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([MorphoDataset.UNK])]
                + [torch.tensor(self.forms.char_vocab.indices(form)) for form in unique_strings], batch_first=True)
            forms_indices = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([unique_string_map[form] for form in sentence]) for sentence in forms], batch_first=True)
            return unique_forms, forms_indices

        def cle_batch_packed(self, forms: list[list[str]]) -> tuple[torch.nn.utils.rnn.PackedSequence,
                                                                    torch.nn.utils.rnn.PackedSequence]:
            """Create a batch suitable for computation of character-level word embeddings.

            This function is very similar to `cle_batch`, but it returns packed sequences instead
            of padded sequences.

            Parameters:
              forms: A batch of sentences, each being a list of string forms.

            Returns:
              unique_forms: A PackedSequence containing each unique form as
                a sequence of character ids.
              forms_indices: A PackedSequence containing for every form from
              the batch its index in `unique_forms`.
            """
            unique_strings = list(set(form for sentence in forms for form in sentence))
            unique_string_map = {form: index + 1 for index, form in enumerate(unique_strings)}
            unique_forms = torch.nn.utils.rnn.pack_sequence(
                [torch.tensor([MorphoDataset.UNK])]
                + [torch.tensor(self.forms.char_vocab.indices(form)) for form in unique_strings], False)
            forms_indices = torch.nn.utils.rnn.pack_sequence(
                [torch.tensor([unique_string_map[form] for form in sentence]) for sentence in forms], False)
            return unique_forms, forms_indices

    def __init__(self, dataset, max_sentences=None):
        """Load the `dataset` dataset, downloading it if necessary.

        Parameters:
          dataset: The name of the dataset, for example `czech_pdt`.
          max_sentences: The maximum number of sentences to load.
        """
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(
                        dataset_file, train=getattr(self, "train", None), max_sentences=max_sentences))

    train: Dataset
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: "MorphoDataset.Factor", predictions: Sequence[str]) -> float:
        """Evaluate the `predictions` against the gold dataset.

        Returns:
          accuracy: The accuracy of the predictions in percentages.
        """
        gold_sentences = gold_dataset.strings

        predicted_sentences, in_sentence = [], False
        for line in predictions:
            line = line.rstrip("\n")
            if not line:
                in_sentence = False
            else:
                if not in_sentence:
                    predicted_sentences.append([])
                    in_sentence = True
                predicted_sentences[-1].append(line)

        if len(predicted_sentences) != len(gold_sentences):
            raise RuntimeError("The predictions contain different number of sentences than gold data: {} vs {}".format(
                len(predicted_sentences), len(gold_sentences)))

        correct, total = 0, 0
        for i, (predicted_sentence, gold_sentence) in enumerate(zip(predicted_sentences, gold_sentences)):
            if len(predicted_sentence) != len(gold_sentence):
                raise RuntimeError("Predicted sentence {} has different number of words than gold: {} vs {}".format(
                    i + 1, len(predicted_sentence), len(gold_sentence)))
            correct += sum(predicted == gold for predicted, gold in zip(predicted_sentence, gold_sentence))
            total += len(predicted_sentence)

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset: "MorphoDataset.Factor", predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          accuracy: The accuracy of the predictions in percentages.
        """
        predictions = predictions_file.readlines()
        return MorphoDataset.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="czech_pdt", type=str, help="The corpus to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
    parser.add_argument("--task", default="tagger", type=str, help="Task to evaluate (tagger/lemmatizer)")
    args = parser.parse_args()

    if args.evaluate:
        gold = getattr(MorphoDataset(args.corpus), args.dataset)
        if args.task == "tagger":
            gold = gold.tags
        elif args.task == "lemmatizer":
            gold = gold.lemmas
        else:
            raise ValueError("Unknown task '{}', valid values are only 'tagger' or 'lemmatizer'".format(args.task))

        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = MorphoDataset.evaluate_file(gold, predictions_file)
        print("{} accuracy: {:.2f}%".format(args.task.title(), accuracy))
