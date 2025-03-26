# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import sys
from typing import Any, Sequence, TextIO, TypedDict
import urllib.request

import torch
import torchvision

from .tfrecord_dataset import TFRecordDataset


class SVHN:
    LABELS: int = 10

    # Type alias for a bounding box -- a list of floats.
    BBox = list[float]

    # The indices of the bounding box coordinates.
    TOP: int = 0
    LEFT: int = 1
    BOTTOM: int = 2
    RIGHT: int = 3

    Element = TypedDict("Element", {"image": torch.Tensor, "classes": torch.Tensor, "bboxes": torch.Tensor})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(TFRecordDataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            super().__init__(path, size, decode_on_demand)

        def __len__(self) -> int:
            return super().__len__()

        def __getitem__(self, index: int) -> "SVHN.Element":
            return super().__getitem__(index)

        def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> "SVHN.Element":
            return {
                "image": torchvision.io.decode_image(
                    data["image"][indices["image"][index]:indices["image"][index + 1]],
                    torchvision.io.ImageReadMode.RGB),
                "classes": data["classes"][indices["classes"][index]:indices["classes"][index + 1]],
                "bboxes": data["bboxes"][indices["bboxes"][index]:indices["bboxes"][index + 1]].view(-1, 4),
            }

    def __init__(self, decode_on_demand: bool = False) -> None:
        for dataset, size in [("train", 10_000), ("dev", 1_267), ("test", 4_535)]:
            path = "svhn.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, self.Dataset(path, size, decode_on_demand))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(
        gold_dataset: Dataset, predictions: Sequence[tuple[list[int], list[BBox]]], iou_threshold: float = 0.5,
    ) -> float:
        def bbox_iou(x: SVHN.BBox, y: SVHN.BBox) -> float:
            def area(bbox: SVHN.BBox) -> float:
                return max(bbox[SVHN.BOTTOM] - bbox[SVHN.TOP], 0) * max(bbox[SVHN.RIGHT] - bbox[SVHN.LEFT], 0)
            intersection = [max(x[SVHN.TOP], y[SVHN.TOP]), max(x[SVHN.LEFT], y[SVHN.LEFT]),
                            min(x[SVHN.BOTTOM], y[SVHN.BOTTOM]), min(x[SVHN.RIGHT], y[SVHN.RIGHT])]
            x_area, y_area, intersection_area = area(x), area(y), area(intersection)
            return intersection_area / (x_area + y_area - intersection_area)

        gold = [(example["classes"].numpy(), example["bboxes"].numpy()) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = 0
        for (gold_classes, gold_bboxes), (prediction_classes, prediction_bboxes) in zip(gold, predictions):
            if len(gold_classes) != len(prediction_classes):
                continue

            used = [False] * len(gold_classes)
            for cls, bbox in zip(prediction_classes, prediction_bboxes):
                best, best_iou = None, None
                for i in range(len(gold_classes)):
                    if used[i] or gold_classes[i] != cls:
                        continue
                    iou = bbox_iou(bbox, gold_bboxes[i])
                    if iou >= iou_threshold and (best is None or iou > best_iou):
                        best, best_iou = i, iou
                if best is None:
                    break
                used[best] = True
            correct += all(used)

        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line in predictions_file:
            values = line.split()
            if len(values) % 5:
                raise RuntimeError("Each prediction must contain multiple of 5 numbers, found {}".format(len(values)))

            predictions.append(([], []))
            for i in range(0, len(values), 5):
                predictions[-1][0].append(int(values[i]))
                predictions[-1][1].append([float(value) for value in values[i + 1:i + 5]])

        return SVHN.evaluate(gold_dataset, predictions)

    # Visualization infrastructure.
    @staticmethod
    def visualize(image: torch.Tensor, labels: list[Any], bboxes: list[BBox], show: bool):
        """Visualize the given image plus recognized objects.

        Parameters:
            image: a torch.Tensor of shape [C, H, W] with dtype torch.uint8
            labels: a list of labels to be shown using the `str` method
            bboxes: a list of `BBox`es (fourtuples TOP, LEFT, BOTTOM, RIGHT)
            show: controls whether to show the figure or return it:
              if `True`, the figure is shown using `plt.show()`;
              if `False`, the `plt.Figure` instance is returned; it can be saved
              to TensorBoard using a the `add_figure` method of a `SummaryWriter`.
        """
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(image.movedim(0, -1).numpy(force=True))
        for label, (top, left, bottom, right) in zip(labels, bboxes):
            label = label.tolist() if isinstance(label, torch.Tensor) else label
            plt.gca().add_patch(plt.Rectangle(
                [left, top], right - left, bottom - top, fill=False, edgecolor=[1, 0, 1], linewidth=2))
            plt.gca().text(left, top, str(label), bbox={"facecolor": [1, 0, 1], "alpha": 0.5},
                           clip_box=plt.gca().clipbox, clip_on=True, ha="left", va="top")

        if show:
            plt.show()
        else:
            return figure


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--visualize", default=None, type=str, help="Prediction file to visualize")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = SVHN.evaluate_file(getattr(SVHN(decode_on_demand=True), args.dataset), predictions_file)
        print("SVHN accuracy: {:.2f}%".format(accuracy))

    if args.visualize:
        with open(args.visualize, "r", encoding="utf-8-sig") as predictions_file:
            for line, example in zip(predictions_file, getattr(SVHN(decode_on_demand=True), args.dataset)):
                values = line.split()
                classes, bboxes = [], []
                for i in range(0, len(values), 5):
                    classes.append(values[i])
                    bboxes.append([float(value) for value in values[i + 1:i + 5]])
                SVHN.visualize(example["image"], classes, bboxes, show=True)
