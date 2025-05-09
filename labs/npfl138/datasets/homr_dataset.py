# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""HOMR dataset contains synthetically generated handwritten music and corresponding annotations.

The input images are grayscale, they have arbitrary height and width, and they
represent a single stave of music. The annotations are “semantic” (the notes are annotated
with their real pitch and duration which depend on current clef and key signature).

A visualization of first 100 training examples is
[available here](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/homr_train.html).
"""
import os
import sys
from typing import Sequence, TextIO, TypedDict
import urllib.request

import torch
import torchvision

from .. import metrics
from .tfrecord_dataset import TFRecordDataset


class HOMRDataset:
    C: int = 1
    """Number of channels in the input images."""

    MARKS: int = 938
    """The number of different marks in the dataset."""
    MARK_NAMES: list[str]  # Set at the bottom of the script for readability
    """The list of mark names in the dataset."""

    Element = TypedDict("Element", {"image": torch.Tensor, "marks": torch.Tensor})
    """The type of a single dataset element."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(TFRecordDataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            super().__init__(path, size, decode_on_demand)

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return super().__len__()

        def __getitem__(self, index: int) -> "HOMRDataset.Element":
            """Return the `index`-th element of the dataset."""
            return super().__getitem__(index)

        def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> "HOMRDataset.Element":
            return {
                "image": torchvision.io.decode_image(
                    data["image"][indices["image"][index]:indices["image"][index + 1]],
                    torchvision.io.ImageReadMode.GRAY),
                "marks": data["marks"][indices["marks"][index]:indices["marks"][index + 1]],
            }

    def __init__(self, decode_on_demand: bool = False) -> None:
        "Load the HOMR dataset, downloading it if necessary."
        for dataset, size in [("train", 51_365), ("dev", 5_027), ("test", 5_023)]:
            path = "homr.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self.URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, self.Dataset(path, size, decode_on_demand))

    train: Dataset
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""

    # The EditDistanceMetric
    EditDistanceMetric = metrics.EditDistance
    """The edit distance metric used for evaluation."""

    # Evaluation infrastructure
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[Sequence[str]]) -> float:
        """Evaluate the `predictions` against the gold dataset.

        Returns:
          edit_distance: The average edit distance of the predictions in percentages.
        """
        gold = [[HOMRDataset.MARK_NAMES[mark] for mark in example["marks"]] for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        return 100 * metrics.EditDistance().update(predictions, gold).compute()

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          edit_distance: The average edit distance of the predictions in percentages.
        """
        predictions = []
        for line in predictions_file:
            predictions.append(line.rstrip("\n").split())
        return HOMRDataset.evaluate(gold_dataset, predictions)


HOMRDataset.MARK_NAMES = [
    "[PAD]", "barline", "clef-C1", "clef-C2", "clef-C3", "clef-C4", "clef-C5", "clef-F3",
    "clef-F4", "clef-G1", "clef-G2", "keySignature-AM", "keySignature-AbM", "keySignature-BM",
    "keySignature-BbM", "keySignature-C#M", "keySignature-CM", "keySignature-DM",
    "keySignature-DbM", "keySignature-EM", "keySignature-EbM", "keySignature-F#M",
    "keySignature-FM", "keySignature-GM", "keySignature-GbM", "note-A#2_eighth", "note-A#2_half",
    "note-A#2_quarter", "note-A#2_quarter.", "note-A#2_sixteenth", "note-A#2_sixteenth.",
    "note-A#3_eighth", "note-A#3_eighth.", "note-A#3_half", "note-A#3_quarter",
    "note-A#3_quarter.", "note-A#3_sixteenth", "note-A#3_sixteenth.", "note-A#3_thirty_second",
    "note-A#3_whole", "note-A#4_eighth", "note-A#4_eighth.", "note-A#4_half", "note-A#4_half.",
    "note-A#4_quarter", "note-A#4_quarter.", "note-A#4_sixteenth", "note-A#4_sixteenth.",
    "note-A#4_thirty_second", "note-A#4_whole", "note-A#4_whole.", "note-A#5_eighth",
    "note-A#5_eighth.", "note-A#5_half", "note-A#5_half.", "note-A#5_quarter", "note-A#5_quarter.",
    "note-A#5_sixteenth", "note-A#5_thirty_second", "note-A1_sixteenth", "note-A2_eighth",
    "note-A2_eighth.", "note-A2_half", "note-A2_half.", "note-A2_quarter", "note-A2_quarter.",
    "note-A2_sixteenth", "note-A2_sixteenth.", "note-A2_thirty_second", "note-A2_whole",
    "note-A2_whole.", "note-A3_eighth", "note-A3_eighth.", "note-A3_half", "note-A3_half.",
    "note-A3_quarter", "note-A3_quarter.", "note-A3_sixteenth", "note-A3_sixteenth.",
    "note-A3_thirty_second", "note-A3_whole", "note-A3_whole.", "note-A4_eighth",
    "note-A4_eighth.", "note-A4_eighth..", "note-A4_half", "note-A4_half.", "note-A4_quarter",
    "note-A4_quarter.", "note-A4_quarter..", "note-A4_sixteenth", "note-A4_sixteenth.",
    "note-A4_thirty_second", "note-A4_whole", "note-A4_whole.", "note-A5_eighth",
    "note-A5_eighth.", "note-A5_eighth..", "note-A5_half", "note-A5_half.", "note-A5_quarter",
    "note-A5_quarter.", "note-A5_quarter..", "note-A5_sixteenth", "note-A5_sixteenth.",
    "note-A5_thirty_second", "note-A5_thirty_second.", "note-A5_whole", "note-A5_whole.",
    "note-Ab2_eighth", "note-Ab2_eighth.", "note-Ab2_half", "note-Ab2_quarter",
    "note-Ab2_quarter.", "note-Ab2_sixteenth", "note-Ab2_thirty_second", "note-Ab2_whole",
    "note-Ab3_eighth", "note-Ab3_eighth.", "note-Ab3_half", "note-Ab3_half.", "note-Ab3_quarter",
    "note-Ab3_quarter.", "note-Ab3_quarter..", "note-Ab3_sixteenth", "note-Ab3_sixteenth.",
    "note-Ab3_thirty_second", "note-Ab3_whole", "note-Ab4_eighth", "note-Ab4_eighth.",
    "note-Ab4_half", "note-Ab4_half.", "note-Ab4_quarter", "note-Ab4_quarter.",
    "note-Ab4_quarter..", "note-Ab4_sixteenth", "note-Ab4_sixteenth.", "note-Ab4_thirty_second",
    "note-Ab4_whole", "note-Ab4_whole.", "note-Ab5_eighth", "note-Ab5_eighth.",
    "note-Ab5_eighth..", "note-Ab5_half", "note-Ab5_half.", "note-Ab5_quarter",
    "note-Ab5_quarter.", "note-Ab5_sixteenth", "note-Ab5_sixteenth.", "note-Ab5_thirty_second",
    "note-Ab5_whole", "note-B#2_eighth", "note-B#2_half", "note-B#2_quarter", "note-B#2_sixteenth",
    "note-B#3_eighth", "note-B#3_eighth.", "note-B#3_half", "note-B#3_half.", "note-B#3_quarter",
    "note-B#3_quarter.", "note-B#3_sixteenth", "note-B#3_thirty_second", "note-B#3_whole",
    "note-B#4_eighth", "note-B#4_eighth.", "note-B#4_half", "note-B#4_half.", "note-B#4_quarter",
    "note-B#4_quarter.", "note-B#4_sixteenth", "note-B#4_sixteenth.", "note-B#4_thirty_second",
    "note-B#4_whole", "note-B#5_eighth", "note-B#5_quarter", "note-B#5_sixteenth",
    "note-B2_eighth", "note-B2_eighth.", "note-B2_half", "note-B2_half.", "note-B2_quarter",
    "note-B2_quarter.", "note-B2_sixteenth", "note-B2_sixteenth.", "note-B2_thirty_second",
    "note-B2_whole", "note-B2_whole.", "note-B3_eighth", "note-B3_eighth.", "note-B3_half",
    "note-B3_half.", "note-B3_quarter", "note-B3_quarter.", "note-B3_quarter..",
    "note-B3_sixteenth", "note-B3_sixteenth.", "note-B3_thirty_second", "note-B3_whole",
    "note-B3_whole.", "note-B4_eighth", "note-B4_eighth.", "note-B4_eighth..", "note-B4_half",
    "note-B4_half.", "note-B4_quarter", "note-B4_quarter.", "note-B4_quarter..",
    "note-B4_sixteenth", "note-B4_sixteenth.", "note-B4_thirty_second", "note-B4_whole",
    "note-B4_whole.", "note-B5_eighth", "note-B5_eighth.", "note-B5_eighth..", "note-B5_half",
    "note-B5_half.", "note-B5_quarter", "note-B5_quarter.", "note-B5_quarter..",
    "note-B5_sixteenth", "note-B5_sixteenth.", "note-B5_thirty_second", "note-B5_whole",
    "note-Bb1_half", "note-Bb2_eighth", "note-Bb2_eighth.", "note-Bb2_half", "note-Bb2_half.",
    "note-Bb2_quarter", "note-Bb2_quarter.", "note-Bb2_sixteenth", "note-Bb2_sixteenth.",
    "note-Bb2_thirty_second", "note-Bb2_whole", "note-Bb2_whole.", "note-Bb3_eighth",
    "note-Bb3_eighth.", "note-Bb3_half", "note-Bb3_half.", "note-Bb3_quarter", "note-Bb3_quarter.",
    "note-Bb3_quarter..", "note-Bb3_sixteenth", "note-Bb3_sixteenth.", "note-Bb3_thirty_second",
    "note-Bb3_whole", "note-Bb3_whole.", "note-Bb4_eighth", "note-Bb4_eighth.",
    "note-Bb4_eighth..", "note-Bb4_half", "note-Bb4_half.", "note-Bb4_quarter",
    "note-Bb4_quarter.", "note-Bb4_quarter..", "note-Bb4_sixteenth", "note-Bb4_sixteenth.",
    "note-Bb4_thirty_second", "note-Bb4_whole", "note-Bb4_whole.", "note-Bb5_eighth",
    "note-Bb5_eighth.", "note-Bb5_half", "note-Bb5_half.", "note-Bb5_quarter", "note-Bb5_quarter.",
    "note-Bb5_quarter..", "note-Bb5_sixteenth", "note-Bb5_sixteenth.", "note-Bb5_thirty_second",
    "note-Bb5_thirty_second.", "note-Bb5_whole", "note-C#2_eighth", "note-C#2_quarter",
    "note-C#2_quarter.", "note-C#2_sixteenth", "note-C#2_whole", "note-C#3_eighth",
    "note-C#3_eighth.", "note-C#3_half", "note-C#3_half.", "note-C#3_quarter", "note-C#3_quarter.",
    "note-C#3_sixteenth", "note-C#3_sixteenth.", "note-C#3_thirty_second", "note-C#3_whole",
    "note-C#4_eighth", "note-C#4_eighth.", "note-C#4_eighth..", "note-C#4_half", "note-C#4_half.",
    "note-C#4_quarter", "note-C#4_quarter.", "note-C#4_sixteenth", "note-C#4_sixteenth.",
    "note-C#4_thirty_second", "note-C#4_whole", "note-C#4_whole.", "note-C#5_eighth",
    "note-C#5_eighth.", "note-C#5_eighth..", "note-C#5_half", "note-C#5_half.", "note-C#5_quarter",
    "note-C#5_quarter.", "note-C#5_quarter..", "note-C#5_sixteenth", "note-C#5_sixteenth.",
    "note-C#5_thirty_second", "note-C#5_whole", "note-C#5_whole.", "note-C#6_eighth",
    "note-C#6_eighth.", "note-C#6_half", "note-C#6_half.", "note-C#6_quarter", "note-C#6_quarter.",
    "note-C#6_quarter..", "note-C#6_sixteenth", "note-C#6_sixteenth.", "note-C#6_thirty_second",
    "note-C2_eighth", "note-C2_eighth.", "note-C2_half", "note-C2_quarter", "note-C2_quarter.",
    "note-C2_sixteenth", "note-C2_thirty_second", "note-C2_whole", "note-C3_eighth",
    "note-C3_eighth.", "note-C3_half", "note-C3_half.", "note-C3_quarter", "note-C3_quarter.",
    "note-C3_sixteenth", "note-C3_sixteenth.", "note-C3_thirty_second", "note-C3_whole",
    "note-C3_whole.", "note-C4_eighth", "note-C4_eighth.", "note-C4_eighth..", "note-C4_half",
    "note-C4_half.", "note-C4_quarter", "note-C4_quarter.", "note-C4_quarter..",
    "note-C4_sixteenth", "note-C4_sixteenth.", "note-C4_thirty_second", "note-C4_whole",
    "note-C4_whole.", "note-C5_eighth", "note-C5_eighth.", "note-C5_eighth..", "note-C5_half",
    "note-C5_half.", "note-C5_quarter", "note-C5_quarter.", "note-C5_quarter..",
    "note-C5_sixteenth", "note-C5_sixteenth.", "note-C5_thirty_second", "note-C5_thirty_second.",
    "note-C5_whole", "note-C5_whole.", "note-C6_eighth", "note-C6_eighth.", "note-C6_eighth..",
    "note-C6_half", "note-C6_half.", "note-C6_half..", "note-C6_quarter", "note-C6_quarter.",
    "note-C6_quarter..", "note-C6_sixteenth", "note-C6_sixteenth.", "note-C6_thirty_second",
    "note-C6_whole", "note-Cb3_eighth", "note-Cb3_quarter", "note-Cb3_thirty_second",
    "note-Cb4_eighth", "note-Cb4_quarter.", "note-Cb4_sixteenth", "note-Cb5_eighth",
    "note-Cb5_eighth.", "note-Cb5_half", "note-Cb5_half.", "note-Cb5_quarter", "note-Cb5_quarter.",
    "note-Cb5_sixteenth", "note-Cb5_thirty_second", "note-Cb5_whole", "note-Cb6_eighth",
    "note-Cb6_half", "note-Cb6_quarter", "note-Cb6_sixteenth", "note-D#2_quarter",
    "note-D#2_sixteenth", "note-D#3_eighth", "note-D#3_eighth.", "note-D#3_half",
    "note-D#3_quarter", "note-D#3_sixteenth", "note-D#3_sixteenth.", "note-D#3_thirty_second",
    "note-D#3_whole", "note-D#4_eighth", "note-D#4_eighth.", "note-D#4_half", "note-D#4_half.",
    "note-D#4_quarter", "note-D#4_quarter.", "note-D#4_sixteenth", "note-D#4_sixteenth.",
    "note-D#4_thirty_second", "note-D#4_whole", "note-D#5_eighth", "note-D#5_eighth.",
    "note-D#5_half", "note-D#5_half.", "note-D#5_quarter", "note-D#5_quarter.",
    "note-D#5_quarter..", "note-D#5_sixteenth", "note-D#5_sixteenth.", "note-D#5_thirty_second",
    "note-D#5_whole", "note-D#6_eighth", "note-D#6_eighth..", "note-D#6_half", "note-D#6_quarter",
    "note-D#6_sixteenth", "note-D#6_thirty_second", "note-D2_eighth", "note-D2_eighth.",
    "note-D2_half", "note-D2_half.", "note-D2_quarter", "note-D2_quarter.", "note-D2_sixteenth",
    "note-D2_thirty_second", "note-D2_whole", "note-D3_eighth", "note-D3_eighth.", "note-D3_half",
    "note-D3_half.", "note-D3_quarter", "note-D3_quarter.", "note-D3_sixteenth",
    "note-D3_sixteenth.", "note-D3_thirty_second", "note-D3_whole", "note-D3_whole.",
    "note-D4_eighth", "note-D4_eighth.", "note-D4_half", "note-D4_half.", "note-D4_quarter",
    "note-D4_quarter.", "note-D4_quarter..", "note-D4_sixteenth", "note-D4_sixteenth.",
    "note-D4_thirty_second", "note-D4_whole", "note-D4_whole.", "note-D5_eighth",
    "note-D5_eighth.", "note-D5_eighth..", "note-D5_half", "note-D5_half.", "note-D5_quarter",
    "note-D5_quarter.", "note-D5_quarter..", "note-D5_sixteenth", "note-D5_sixteenth.",
    "note-D5_thirty_second", "note-D5_thirty_second.", "note-D5_whole", "note-D5_whole.",
    "note-D6_eighth", "note-D6_eighth.", "note-D6_eighth..", "note-D6_half", "note-D6_half.",
    "note-D6_half..", "note-D6_quarter", "note-D6_quarter.", "note-D6_quarter..",
    "note-D6_sixteenth", "note-D6_sixteenth.", "note-D6_thirty_second", "note-D6_whole",
    "note-D6_whole.", "note-Db3_eighth", "note-Db3_half", "note-Db3_half.", "note-Db3_quarter",
    "note-Db3_quarter.", "note-Db3_thirty_second", "note-Db4_eighth", "note-Db4_eighth.",
    "note-Db4_half", "note-Db4_half.", "note-Db4_quarter", "note-Db4_quarter.",
    "note-Db4_sixteenth", "note-Db4_sixteenth.", "note-Db4_thirty_second", "note-Db4_whole",
    "note-Db5_eighth", "note-Db5_eighth.", "note-Db5_half", "note-Db5_half.", "note-Db5_quarter",
    "note-Db5_quarter.", "note-Db5_quarter..", "note-Db5_sixteenth", "note-Db5_sixteenth.",
    "note-Db5_thirty_second", "note-Db5_whole", "note-Db5_whole.", "note-Db6_eighth",
    "note-Db6_eighth.", "note-Db6_half", "note-Db6_quarter", "note-Db6_quarter.",
    "note-Db6_sixteenth", "note-Db6_thirty_second", "note-E#3_eighth", "note-E#3_eighth.",
    "note-E#3_half", "note-E#3_sixteenth", "note-E#4_eighth", "note-E#4_eighth.", "note-E#4_half",
    "note-E#4_quarter", "note-E#4_quarter.", "note-E#4_sixteenth", "note-E#4_whole",
    "note-E#4_whole.", "note-E#5_eighth", "note-E#5_eighth.", "note-E#5_half", "note-E#5_half.",
    "note-E#5_quarter", "note-E#5_quarter.", "note-E#5_sixteenth", "note-E#5_sixteenth.",
    "note-E#5_thirty_second", "note-E#6_sixteenth", "note-E2_eighth", "note-E2_eighth.",
    "note-E2_half", "note-E2_half.", "note-E2_quarter", "note-E2_sixteenth",
    "note-E2_thirty_second", "note-E2_whole", "note-E3_eighth", "note-E3_eighth.", "note-E3_half",
    "note-E3_half.", "note-E3_quarter", "note-E3_quarter.", "note-E3_sixteenth",
    "note-E3_sixteenth.", "note-E3_thirty_second", "note-E3_whole", "note-E3_whole.",
    "note-E4_eighth", "note-E4_eighth.", "note-E4_eighth..", "note-E4_half", "note-E4_half.",
    "note-E4_quarter", "note-E4_quarter.", "note-E4_quarter..", "note-E4_sixteenth",
    "note-E4_sixteenth.", "note-E4_thirty_second", "note-E4_whole", "note-E4_whole.",
    "note-E5_eighth", "note-E5_eighth.", "note-E5_eighth..", "note-E5_half", "note-E5_half.",
    "note-E5_half..", "note-E5_quarter", "note-E5_quarter.", "note-E5_quarter..",
    "note-E5_sixteenth", "note-E5_sixteenth.", "note-E5_thirty_second", "note-E5_whole",
    "note-E5_whole.", "note-E6_eighth", "note-E6_eighth.", "note-E6_eighth..", "note-E6_half",
    "note-E6_half.", "note-E6_quarter", "note-E6_quarter.", "note-E6_sixteenth",
    "note-E6_sixteenth.", "note-E6_thirty_second", "note-Eb2_eighth", "note-Eb2_half",
    "note-Eb2_quarter", "note-Eb2_quarter.", "note-Eb2_sixteenth", "note-Eb2_sixteenth.",
    "note-Eb2_thirty_second", "note-Eb2_whole", "note-Eb3_eighth", "note-Eb3_eighth.",
    "note-Eb3_half", "note-Eb3_half.", "note-Eb3_quarter", "note-Eb3_quarter.",
    "note-Eb3_sixteenth", "note-Eb3_sixteenth.", "note-Eb3_thirty_second", "note-Eb3_whole",
    "note-Eb3_whole.", "note-Eb4_eighth", "note-Eb4_eighth.", "note-Eb4_half", "note-Eb4_half.",
    "note-Eb4_quarter", "note-Eb4_quarter.", "note-Eb4_quarter..", "note-Eb4_sixteenth",
    "note-Eb4_sixteenth.", "note-Eb4_thirty_second", "note-Eb4_whole", "note-Eb4_whole.",
    "note-Eb5_eighth", "note-Eb5_eighth.", "note-Eb5_eighth..", "note-Eb5_half", "note-Eb5_half.",
    "note-Eb5_quarter", "note-Eb5_quarter.", "note-Eb5_quarter..", "note-Eb5_sixteenth",
    "note-Eb5_sixteenth.", "note-Eb5_thirty_second", "note-Eb5_whole", "note-Eb5_whole.",
    "note-Eb6_eighth", "note-Eb6_eighth.", "note-Eb6_eighth..", "note-Eb6_half", "note-Eb6_half.",
    "note-Eb6_quarter", "note-Eb6_quarter.", "note-Eb6_sixteenth", "note-Eb6_sixteenth.",
    "note-Eb6_thirty_second", "note-F#2_eighth", "note-F#2_eighth.", "note-F#2_half",
    "note-F#2_half.", "note-F#2_quarter", "note-F#2_quarter.", "note-F#2_sixteenth",
    "note-F#2_whole", "note-F#3_eighth", "note-F#3_eighth.", "note-F#3_half", "note-F#3_half.",
    "note-F#3_quarter", "note-F#3_quarter.", "note-F#3_sixteenth", "note-F#3_sixteenth.",
    "note-F#3_thirty_second", "note-F#3_whole", "note-F#3_whole.", "note-F#4_eighth",
    "note-F#4_eighth.", "note-F#4_half", "note-F#4_half.", "note-F#4_quarter", "note-F#4_quarter.",
    "note-F#4_quarter..", "note-F#4_sixteenth", "note-F#4_sixteenth.", "note-F#4_thirty_second",
    "note-F#4_whole", "note-F#4_whole.", "note-F#5_eighth", "note-F#5_eighth.", "note-F#5_half",
    "note-F#5_half.", "note-F#5_quarter", "note-F#5_quarter.", "note-F#5_quarter..",
    "note-F#5_sixteenth", "note-F#5_sixteenth.", "note-F#5_thirty_second", "note-F#5_whole",
    "note-F#5_whole.", "note-F#6_eighth", "note-F#6_eighth.", "note-F#6_half", "note-F#6_quarter",
    "note-F#6_quarter.", "note-F#6_sixteenth", "note-F#6_thirty_second", "note-F#6_whole",
    "note-F2_eighth", "note-F2_eighth.", "note-F2_half", "note-F2_half.", "note-F2_quarter",
    "note-F2_quarter.", "note-F2_quarter..", "note-F2_sixteenth", "note-F2_sixteenth.",
    "note-F2_thirty_second", "note-F2_whole", "note-F2_whole.", "note-F3_eighth",
    "note-F3_eighth.", "note-F3_half", "note-F3_half.", "note-F3_quarter", "note-F3_quarter.",
    "note-F3_quarter..", "note-F3_sixteenth", "note-F3_sixteenth.", "note-F3_thirty_second",
    "note-F3_whole", "note-F3_whole.", "note-F4_eighth", "note-F4_eighth.", "note-F4_eighth..",
    "note-F4_half", "note-F4_half.", "note-F4_quarter", "note-F4_quarter.", "note-F4_quarter..",
    "note-F4_sixteenth", "note-F4_sixteenth.", "note-F4_thirty_second", "note-F4_whole",
    "note-F4_whole.", "note-F5_eighth", "note-F5_eighth.", "note-F5_half", "note-F5_half.",
    "note-F5_quarter", "note-F5_quarter.", "note-F5_quarter..", "note-F5_sixteenth",
    "note-F5_sixteenth.", "note-F5_thirty_second", "note-F5_whole", "note-F5_whole.",
    "note-F6_eighth", "note-F6_eighth.", "note-F6_half", "note-F6_half.", "note-F6_quarter",
    "note-F6_quarter.", "note-F6_sixteenth", "note-F6_sixteenth.", "note-F6_thirty_second",
    "note-Fb3_half", "note-Fb3_sixteenth", "note-Fb3_thirty_second", "note-Fb4_eighth",
    "note-Fb4_quarter.", "note-Fb4_sixteenth", "note-Fb4_thirty_second", "note-Fb5_eighth",
    "note-Fb5_half", "note-G#2_eighth", "note-G#2_eighth.", "note-G#2_half", "note-G#2_half.",
    "note-G#2_quarter", "note-G#2_quarter.", "note-G#2_sixteenth", "note-G#3_eighth",
    "note-G#3_eighth.", "note-G#3_half", "note-G#3_half.", "note-G#3_quarter", "note-G#3_quarter.",
    "note-G#3_sixteenth", "note-G#3_sixteenth.", "note-G#3_thirty_second", "note-G#3_whole",
    "note-G#4_eighth", "note-G#4_eighth.", "note-G#4_eighth..", "note-G#4_half", "note-G#4_half.",
    "note-G#4_quarter", "note-G#4_quarter.", "note-G#4_sixteenth", "note-G#4_sixteenth.",
    "note-G#4_thirty_second", "note-G#4_whole", "note-G#4_whole.", "note-G#5_eighth",
    "note-G#5_eighth.", "note-G#5_half", "note-G#5_half.", "note-G#5_quarter", "note-G#5_quarter.",
    "note-G#5_quarter..", "note-G#5_sixteenth", "note-G#5_sixteenth.", "note-G#5_thirty_second",
    "note-G#5_whole", "note-G#5_whole.", "note-G2_eighth", "note-G2_eighth.", "note-G2_half",
    "note-G2_half.", "note-G2_quarter", "note-G2_quarter.", "note-G2_sixteenth",
    "note-G2_sixteenth.", "note-G2_thirty_second", "note-G2_whole", "note-G2_whole.",
    "note-G3_eighth", "note-G3_eighth.", "note-G3_half", "note-G3_half.", "note-G3_quarter",
    "note-G3_quarter.", "note-G3_quarter..", "note-G3_sixteenth", "note-G3_sixteenth.",
    "note-G3_thirty_second", "note-G3_whole", "note-G3_whole.", "note-G4_eighth",
    "note-G4_eighth.", "note-G4_eighth..", "note-G4_half", "note-G4_half.", "note-G4_quarter",
    "note-G4_quarter.", "note-G4_quarter..", "note-G4_sixteenth", "note-G4_sixteenth.",
    "note-G4_thirty_second", "note-G4_whole", "note-G4_whole.", "note-G5_eighth",
    "note-G5_eighth.", "note-G5_eighth..", "note-G5_half", "note-G5_half.", "note-G5_half..",
    "note-G5_quarter", "note-G5_quarter.", "note-G5_quarter..", "note-G5_sixteenth",
    "note-G5_sixteenth.", "note-G5_thirty_second", "note-G5_whole", "note-G5_whole.",
    "note-Gb3_eighth", "note-Gb3_eighth.", "note-Gb3_half", "note-Gb3_quarter",
    "note-Gb3_quarter..", "note-Gb3_thirty_second", "note-Gb4_eighth", "note-Gb4_eighth.",
    "note-Gb4_half", "note-Gb4_half.", "note-Gb4_quarter", "note-Gb4_quarter.",
    "note-Gb4_sixteenth", "note-Gb4_thirty_second", "note-Gb4_whole", "note-Gb5_eighth",
    "note-Gb5_eighth.", "note-Gb5_half", "note-Gb5_half.", "note-Gb5_quarter",
    "note-Gb5_quarter.", "note-Gb5_sixteenth", "note-Gb5_whole", "rest-eighth", "rest-eighth.",
    "rest-half", "rest-half.", "rest-quadruple_whole", "rest-quarter", "rest-quarter.",
    "rest-quarter..", "rest-sixteenth", "rest-sixteenth.", "rest-whole", "tie",
    "timeSignature-1/2", "timeSignature-1/4", "timeSignature-2/1", "timeSignature-2/2",
    "timeSignature-2/3", "timeSignature-2/4", "timeSignature-2/8", "timeSignature-3/1",
    "timeSignature-3/2", "timeSignature-3/4", "timeSignature-3/6", "timeSignature-3/8",
    "timeSignature-4/2", "timeSignature-4/4", "timeSignature-4/8", "timeSignature-5/4",
    "timeSignature-5/8", "timeSignature-6/2", "timeSignature-6/4", "timeSignature-6/8",
    "timeSignature-7/4", "timeSignature-8/2", "timeSignature-8/4", "timeSignature-8/8",
    "timeSignature-9/4", "timeSignature-9/8", "timeSignature-C", "timeSignature-C/",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            edit_distance = HOMRDataset.evaluate_file(
                getattr(HOMRDataset(decode_on_demand=True), args.dataset), predictions_file)
        print("HOMR edit distance: {:.3f}%".format(edit_distance))
