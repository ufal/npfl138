import array
import os
import sys
import struct
from typing import Any, Callable, Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch
import torchvision


class SVHN:
    LABELS: int = 10

    # Type alias for a bounding box -- a list of floats.
    BBox = list[float]

    # The indices of the bounding box coordinates.
    TOP: int = 0
    LEFT: int = 1
    BOTTOM: int = 2
    RIGHT: int = 3

    Element = TypedDict("Element", {"image": torch.Tensor, "classes": np.ndarray, "bboxes": np.ndarray})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            self._size = size

            arrays, indices = SVHN._load_data(path, size)
            if decode_on_demand:
                self._data, self._arrays, self._indices = None, arrays, indices
            else:
                self._data = [self._decode(arrays, indices, i) for i in range(size)]

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> "SVHN.Element":
            if self._data:
                return self._data[index]
            return self._decode(self._arrays, self._indices, index)

        def transform(self, transform: Callable[["SVHN.Element"], Any]) -> "SVHN.TransformedDataset":
            return SVHN.TransformedDataset(self, transform)

        def _decode(self, data: dict, indices: dict, index: int) -> "SVHN.Element":
            return {
                "image": torchvision.io.decode_image(
                    torch.frombuffer(data["image"], dtype=torch.uint8, offset=indices["image"][:-1][index],
                                     count=indices["image"][1:][index] - indices["image"][:-1][index]),
                    torchvision.io.ImageReadMode.RGB).permute(1, 2, 0),
                "classes": np.frombuffer(
                    data["classes"], dtype=np.int64, offset=indices["classes"][:-1][index] << 3,
                    count=indices["classes"][1:][index] - indices["classes"][:-1][index]),
                "bboxes": np.frombuffer(
                    data["bboxes"], dtype=np.int64, offset=indices["bboxes"][:-1][index] << 3,
                    count=indices["bboxes"][1:][index] - indices["bboxes"][:-1][index]).reshape(-1, 4),
            }

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "SVHN.TransformedDataset":
            return SVHN.TransformedDataset(self, transform)

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

    # TFRecord loading
    @staticmethod
    def _load_data(path: str, items: int) -> tuple[dict[str, array.array], dict[str, array.array]]:
        def get_value() -> np.int64:
            nonlocal data, offset
            value = np.int64(data[offset] & 0x7F); start = offset; offset += 1
            while data[offset - 1] & 0x80:
                value |= (data[offset] & 0x7F) << (7 * (offset - start)); offset += 1
            return value

        def get_value_of_kind(kind: int) -> np.int64:
            nonlocal data, offset
            assert data[offset] == kind; offset += 1
            return get_value()

        arrays, indices = {}, {}
        with open(path, "rb") as file:
            for _ in range(items):
                length = file.read(8); assert len(length) == 8
                length, = struct.unpack("<Q", length)
                assert len(file.read(4)) == 4
                data = file.read(length); assert len(data) == length
                assert len(file.read(4)) == 4

                offset = 0
                length = get_value_of_kind(0x0A)
                assert len(data) - offset == length
                while offset < len(data):
                    get_value_of_kind(0x0A)
                    length = get_value_of_kind(0x0A)
                    key = data[offset:offset + length].decode("utf-8"); offset += length
                    get_value_of_kind(0x12)
                    if key not in arrays:
                        arrays[key] = array.array({0x0A: "B", 0x1A: "Q", 0x12: "f"}.get(data[offset], "B"))
                        indices[key] = array.array("L", [0])

                    if data[offset] == 0x0A:
                        length = get_value_of_kind(0x0A) and get_value_of_kind(0x0A)
                        arrays[key].frombytes(data[offset:offset + length]); offset += length
                    elif data[offset] == 0x1A:
                        length = get_value_of_kind(0x1A) and get_value_of_kind(0x0A)
                        target_offset = offset + length
                        while offset < target_offset:
                            arrays[key].append(get_value())
                    elif data[offset] == 0x12:
                        length = get_value_of_kind(0x12) and get_value_of_kind(0x0A)
                        arrays[key].frombytes(np.frombuffer(
                            data, np.dtype("<f4"), length >> 2, offset).astype(np.float32).tobytes()); offset += length
                    else:
                        raise ValueError("Unsupported data tag {}".format(data[offset]))
                    indices[key].append(len(arrays[key]))
        return arrays, indices

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(
        gold_dataset: "SVHN.Dataset", predictions: Sequence[tuple[list[int], list[BBox]]], iou_threshold: float = 0.5,
    ) -> float:
        def bbox_iou(x: SVHN.BBox, y: SVHN.BBox) -> float:
            def area(bbox: SVHN.BBox) -> float:
                return max(bbox[SVHN.BOTTOM] - bbox[SVHN.TOP], 0) * max(bbox[SVHN.RIGHT] - bbox[SVHN.LEFT], 0)
            intersection = [max(x[SVHN.TOP], y[SVHN.TOP]), max(x[SVHN.LEFT], y[SVHN.LEFT]),
                            min(x[SVHN.BOTTOM], y[SVHN.BOTTOM]), min(x[SVHN.RIGHT], y[SVHN.RIGHT])]
            x_area, y_area, intersection_area = area(x), area(y), area(intersection)
            return intersection_area / (x_area + y_area - intersection_area)

        gold = [(np.array(example["classes"]), np.array(example["bboxes"])) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = 0
        for (gold_classes, gold_bboxes), (prediction_classes, prediction_bboxes) in zip(gold, predictions):
            if len(gold_classes) != len(prediction_classes):
                continue

            used = [False] * len(gold_classes)
            for cls, bbox in zip(prediction_classes, prediction_bboxes):
                best = None
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
    def visualize(image: np.ndarray, labels: list[Any], bboxes: list[BBox], show: bool):
        """Visualize the given image plus recognized objects.

        Arguments:
        - `image` is NumPy input image with pixels in range [0-255];
        - `labels` is a list of labels to be shown using the `str` method;
        - `bboxes` is a list of `BBox`es (fourtuples TOP, LEFT, BOTTOM, RIGHT);
        - `show` controls whether to show the figure or return it:
          - if `True`, the figure is shown using `plt.show()`;
          - if `False`, the `plt.Figure` instance is returned; it can be saved
            to TensorBoard using a the `add_figure` method of a `SummaryWriter`.
        """
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(4, 4))
        plt.axis("off")
        plt.imshow(np.asarray(image, np.uint8))
        for label, (top, left, bottom, right) in zip(labels, bboxes):
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
