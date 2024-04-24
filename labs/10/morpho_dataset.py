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


# Loads a morphological dataset in a vertical format.
# - The data consists of three datasets
#   - `train`
#   - `dev`
#   - `test`
# - Each dataset is a `torch.utils.data.Dataset` providing
#   - `__len__`: number of sentences in the dataset
#   - `__getitem__`: return the requested sentence as an `Element`
#     instance, which is a dictionary with keys "forms"/"lemmas"/"tags",
#     each being a list of strings
#   - `forms`, `lemmas`, `tags`: instances of type `Factor` containing
#     the following fields:
#     - `strings`: a Python list containing input sentences, each being
#       a list of strings (forms/lemmas/tags)
#     - `word_vocab`: a `Vocabulary` object capable of mapping words to
#       indices. It is constructed on the train set and shared by the dev
#       and test sets
#     - `char_vocab`: a `Vocabulary` object capable of mapping characters
#       to  indices. It is constructed on the train set and shared by the dev
#       and test sets
#   - `cle_batch`: a method for creating inputs for character-level embeddings.
#     It takes a list of sentences, each being a list of string forms, and produces
#     a tuple of two tensors:
#     - `unique_forms` with shape `[num_unique_forms, max_form_length]` containing
#       each unique form as a sequence of character ids
#     - `forms_indices` with shape `[num_sentences, max_sentence_length]`
#       containing for every form its index in `unique_forms`
class MorphoDataset:
    PAD: int = 0
    UNK: int = 1
    BOW: int = 2
    EOW: int = 3

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    Element = TypedDict("Element", {"forms": list[str], "lemmas": list[str], "tags": list[str]})

    class Factor:
        word_vocab: Vocabulary
        char_vocab: Vocabulary
        strings: list[list[str]]

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
            return self._factors[0]

        @property
        def lemmas(self) -> "MorphoDataset.Factor":
            return self._factors[1]

        @property
        def tags(self) -> "MorphoDataset.Factor":
            return self._factors[2]

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> "MorphoDataset.Element":
            return {"forms": self.forms.strings[index],
                    "lemmas": self.lemmas.strings[index],
                    "tags": self.tags.strings[index]}

        def transform(self, transform: Callable[["MorphoDataset.Element"], Any]) -> "MorphoDataset.TransformedDataset":
            return MorphoDataset.TransformedDataset(self, transform)

        def cle_batch(self, forms: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
            unique_strings = list(set(form for sentence in forms for form in sentence))
            unique_string_map = {form: index + 1 for index, form in enumerate(unique_strings)}
            unique_forms = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([MorphoDataset.UNK])]
                + [torch.tensor(self.forms.char_vocab.indices(form)) for form in unique_strings], batch_first=True)
            forms_indices = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([unique_string_map[form] for form in sentence]) for sentence in forms], batch_first=True)
            return unique_forms, forms_indices

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "MorphoDataset.TransformedDataset":
            return MorphoDataset.TransformedDataset(self, transform)

    def __init__(self, dataset, max_sentences=None):
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(
                        dataset_file, train=self.train if dataset != "train" else None,
                        max_sentences=max_sentences))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: "MorphoDataset.Factor", predictions: Sequence[str]) -> float:
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
