# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The `ReadingComprehensionDataset` class represents a reading comprehension dataset.

- Loads a reading comprehension data.
- The data consists of three datasets:
    - `train`
    - `dev`
    - `test`
- Each dataset contains a list of paragraphs in the `paragraphs` field.
- Each paragraph is a dictionary containing the following:
    - `context`: text
    - `qas`: list of questions and answers, each a dictionary with:
        - `question`: text of the question
        - `answers`: a list of answers, each answer a dictionary containing:
            - `text`: answer test as string, exactly as appearing in the context
            - `start`: character offset of the answer text in the context
"""
import os
import sys
from typing import BinaryIO, Sequence, TextIO, TypedDict
import urllib.request
import zipfile


class ReadingComprehensionDataset:
    Paragraph = TypedDict("Paragraph", {"context": str, "qas": list[TypedDict("QA", {
        "question": str, "answers": list[TypedDict("Answer", {"text": str, "start": int})]})]})
    """The type of a single Paragraph containing possibly several questions and corresponding answers."""
    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset:
        def __init__(self, data_file: BinaryIO) -> None:
            # Load the data
            self._paragraphs = []
            in_paragraph = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_paragraph:
                        self._paragraphs.append({"context": line, "qas": []})
                        in_paragraph = True
                    else:
                        question, *qas = line.split("\t")
                        assert len(qas) % 2 == 0

                        self._paragraphs[-1]["qas"].append({
                            "question": question,
                            "answers": [
                                {"text": qas[i], "start": int(qas[i + 1])} for i in range(0, len(qas), 2)]})
                else:
                    in_paragraph = False

        @property
        def paragraphs(self) -> list["ReadingComprehensionDataset.Paragraph"]:
            """The paragraphs in this dataset."""
            return self._paragraphs

    def __init__(self, name: str = "reading_comprehension") -> None:
        """Load the dataset, downloading it if necessary."""
        path = "{}.zip".format(name)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file))

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
        gold = [qa["answers"] for paragraph in gold_dataset.paragraphs for qa in paragraph["qas"]]
        if len(predictions) != len(gold):
            raise RuntimeError("The predictions contain different number of answers than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct, total = 0, 0
        for prediction, gold_answers in zip(predictions, gold):
            correct += any(prediction == gold_answer["text"] for gold_answer in gold_answers)
            total += 1

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          accuracy: The accuracy of the predictions in percentages.
        """
        predictions = [answer.strip() for answer in predictions_file]
        return ReadingComprehensionDataset.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="reading_comprehension", type=str, help="The corpus to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = ReadingComprehensionDataset.evaluate_file(
                getattr(ReadingComprehensionDataset(args.corpus), args.dataset), predictions_file)
        print("Reading comprehension accuracy: {:.2f}%".format(accuracy))
