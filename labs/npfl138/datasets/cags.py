# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
import sys
from typing import Sequence, TextIO, TypedDict
import urllib.request

import torch
import torchvision

from .tfrecord_dataset import TFRecordDataset


class CAGS:
    C: int = 3
    H: int = 224
    W: int = 224
    LABELS: int = 34
    LABEL_NAMES: list[str] = [
        # Cats
        "Abyssinian", "Bengal", "Bombay", "British_Shorthair", "Egyptian_Mau",
        "Maine_Coon", "Russian_Blue", "Siamese", "Sphynx",
        # Dogs
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]
    Element = TypedDict("Element", {"image": torch.Tensor, "mask": torch.Tensor, "label": torch.Tensor})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(TFRecordDataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            super().__init__(path, size, decode_on_demand)

        def __len__(self) -> int:
            return super().__len__()

        def __getitem__(self, index: int) -> "CAGS.Element":
            return super().__getitem__(index)

        def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> "CAGS.Element":
            return {
                "image": torchvision.io.decode_image(
                    data["image"][indices["image"][index]:indices["image"][index + 1]],
                    torchvision.io.ImageReadMode.RGB),
                "mask": torchvision.io.decode_image(
                    data["mask"][indices["mask"][index]:indices["mask"][index + 1]],
                    torchvision.io.ImageReadMode.GRAY).to(dtype=torch.float32).div(255),
                "label": data["label"][index],
            }

    def __init__(self, decode_on_demand: bool = False) -> None:
        for dataset, size in [("train", 2_142), ("dev", 306), ("test", 612)]:
            path = "cags.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, self.Dataset(path, size, decode_on_demand))

    train: Dataset
    dev: Dataset
    test: Dataset

    # The MaskIoUMetric
    class MaskIoUMetric(torch.nn.Module):
        """MaskIoUMetric computes IoU for CAGS dataset masks predicted by binary classification"""
        def __init__(self):
            super().__init__()
            self.register_buffer("iou", torch.tensor(0.0, dtype=torch.float32), persistent=False)
            self.register_buffer("count", torch.tensor(0, dtype=torch.int64), persistent=False)

        def reset(self):
            self.iou.zero_()
            self.count.zero_()

        def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
            y_pred_mask = (y_pred.detach() >= 0.5).reshape([-1, CAGS.H * CAGS.W])
            y_true_mask = (y_true.detach() >= 0.5).reshape([-1, CAGS.H * CAGS.W])

            intersection = torch.logical_and(y_pred_mask, y_true_mask).float().sum(dim=1)
            union = torch.logical_or(y_pred_mask, y_true_mask).float().sum(dim=1)
            iou = torch.where(union == 0, 1., intersection / union)

            self.iou += iou.sum()
            self.count += iou.shape[0]

        def compute(self) -> torch.Tensor:
            return self.iou / self.count

    # Evaluation infrastructure.
    @staticmethod
    def evaluate_classification(gold_dataset: Dataset, predictions: Sequence[int]) -> float:
        gold = [int(example["label"]) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_classification_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [int(line) for line in predictions_file]
        return CAGS.evaluate_classification(gold_dataset, predictions)

    @staticmethod
    def evaluate_segmentation(gold_dataset: Dataset, predictions: Sequence[torch.Tensor]) -> float:
        gold = [example["mask"] for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        iou = CAGS.MaskIoUMetric()
        for i in range(len(gold)):
            iou.update(gold[i], predictions[i])

        return 100 * iou.compute()

    @staticmethod
    def load_segmentation_file(predictions_file: TextIO) -> list[torch.Tensor]:
        predictions = []
        for line in predictions_file:
            runs = [int(run) for run in line.split()]
            assert sum(runs) == CAGS.H * CAGS.W

            offset = 0
            predictions.append(torch.zeros(CAGS.H * CAGS.W, dtype=torch.float32))
            for i, run in enumerate(runs):
                predictions[-1][offset:offset + run] = i % 2
                offset += run
        return predictions

    @staticmethod
    def evaluate_segmentation_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        return CAGS.evaluate_segmentation(gold_dataset, CAGS.load_segmentation_file(predictions_file))

    @staticmethod
    def visualize(image: torch.Tensor, mask: torch.Tensor, show: bool):
        """Visualize the given image plus predicted mask.

        Parameters:
            image: A torch.Tensor of shape [C, H, W] with dtype torch.uint8
            mask: A torch.Tensor with H * W float values in [0, 1]
            show: controls whether to show the figure or return it:
              if `True`, the figure is shown using `plt.show()`;
              if `False`, the `plt.Figure` instance is returned; it can be saved
              to TensorBoard using a the `add_figure` method of a `SummaryWriter`.
        """
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(10, 4))
        plt.axis("off")
        byte_mask = mask.reshape([CAGS.H, CAGS.W]).to(dtype=torch.uint8)
        visualization = torch.zeros([3, CAGS.H, 3 * CAGS.W], dtype=torch.uint8)
        visualization[:, :, :CAGS.W] = image
        visualization[:, :, CAGS.W:2 * CAGS.W] = 255 * byte_mask
        visualization[:, :, 2 * CAGS.W:] = image * byte_mask + 255 * (1 - byte_mask)
        plt.imshow(visualization.movedim(0, -1).numpy(force=True))
        if show:
            plt.show()
        else:
            return figure


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--evaluate_classification", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--evaluate_segmentation", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--visualize_segmentation", default=None, type=str, help="Prediction file to visualize")
    args = parser.parse_args()

    gold_dataset = getattr(CAGS(decode_on_demand=True), args.dataset)
    if args.evaluate_classification:
        with open(args.evaluate_classification, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = CAGS.evaluate_classification_file(gold_dataset, predictions_file)
        print("CAGS accuracy: {:.2f}%".format(accuracy))

    if args.evaluate_segmentation:
        with open(args.evaluate_segmentation, "r", encoding="utf-8-sig") as predictions_file:
            iou = CAGS.evaluate_segmentation_file(gold_dataset, predictions_file)
        print("CAGS IoU: {:.2f}%".format(iou))

    if args.visualize_segmentation:
        with open(args.visualize_segmentation, "r", encoding="utf-8-sig") as predictions_file:
            predictions = CAGS.load_segmentation_file(predictions_file)
        for example, prediction in zip(gold_dataset, predictions):
            CAGS.visualize(example["image"], prediction, show=True)
