import os
import sys
import struct
from typing import Any, Callable, Sequence, TextIO
import urllib.request
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torchvision


class CAGS:
    H: int = 224
    W: int = 224
    C: int = 3
    LABELS: list[str] = [
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

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, size: int) -> None:
            self._path = path
            self._data = None
            self._size = size

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            if self._data is None:
                self._data = []
                for entry in CAGS._load_data(self._path, self._size):
                    entry["image"] = torchvision.io.decode_image(
                        torch.from_numpy(entry["image"]), torchvision.io.ImageReadMode.RGB).permute(1, 2, 0)
                    entry["mask"] = (torchvision.io.decode_image(torch.from_numpy(entry["mask"])).to(
                        dtype=torch.float32) / 255).permute(1, 2, 0)
                    entry["label"] = torch.tensor(entry["label"][0])
                    self._data.append(entry)
            return self._data[index]

        def transform(self, transform: Callable[[dict[str, torch.Tensor]], Any]) -> torch.utils.data.Dataset:
            return CAGS.TransformedDataset(self, transform)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: "Dataset", transform: Callable[[dict[str, torch.Tensor]], Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return self._dataset._size

        def __getitem__(self, index: int) -> Any:
            return self._transform(self._dataset[index])

    def __init__(self) -> None:
        for dataset, size in [("train", 2_142), ("dev", 306), ("test", 612)]:
            path = "cags.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, self.Dataset(path, size))

    train: Dataset
    dev: Dataset
    test: Dataset

    # TFRecord loading
    @staticmethod
    def _load_data(path: str, items: int) -> list[dict[str, Any]]:
        def get_value() -> int:
            nonlocal data, offset
            value = np.int64(data[offset] & 0x7F); start = offset; offset += 1
            while data[offset - 1] & 0x80:
                value |= (data[offset] & 0x7F) << (7 * (offset - start)); offset += 1
            return value

        def get_value_of_kind(kind: int) -> int:
            nonlocal data, offset
            assert data[offset] == kind; offset += 1
            return get_value()

        entries = []
        with open(path, "rb") as file:
            while len(entries) < items:
                entries.append({})

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
                    if data[offset] == 0x0A:
                        get_value_of_kind(0x0A)
                        length = get_value_of_kind(0x0A)
                        entries[-1][key] = np.frombuffer(data, np.uint8, length, offset).copy(); offset += length
                    elif data[offset] == 0x1A:
                        get_value_of_kind(0x1A)
                        length = get_value_of_kind(0x0A)
                        values, target_offset = [], offset + length
                        while offset < target_offset:
                            values.append(get_value())
                        entries[-1][key] = np.array(values, dtype=np.int64)
                    elif data[offset] == 0x12:
                        get_value_of_kind(0x12)
                        length = get_value_of_kind(0x0A)
                        entries[-1][key] = np.frombuffer(
                            data, np.dtype("<f4"), length >> 2, offset).astype(np.float32).copy(); offset += length
                    else:
                        raise ValueError("Unsupported data tag {}".format(data[offset]))
        return entries

    # Keras IoU metric
    class MaskIoUMetric(keras.metrics.Mean):
        """MaskIoUMetric computes IoU for CAGS dataset masks predicted by binary classification"""
        def __init__(self, name="iou", dtype=None) -> None:
            super().__init__(name, dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true_mask = keras.ops.reshape(y_true >= 0.5, [-1, CAGS.H * CAGS.W])
            y_pred_mask = keras.ops.reshape(y_pred >= 0.5, [-1, CAGS.H * CAGS.W])

            intersection_mask = keras.ops.logical_and(y_true_mask, y_pred_mask)
            union_mask = keras.ops.logical_or(y_true_mask, y_pred_mask)

            intersection = keras.ops.sum(keras.ops.cast(intersection_mask, "float32"), axis=1)
            union = keras.ops.sum(keras.ops.cast(union_mask, "float32"), axis=1)
            iou = keras.ops.where(union == 0, 1., intersection / union)
            return super().update_state(iou, sample_weight)

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
            iou(gold[i], predictions[i])

        return 100 * iou.result()

    @staticmethod
    def evaluate_segmentation_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line in predictions_file:
            runs = [int(run) for run in line.split()]
            assert sum(runs) == CAGS.H * CAGS.W

            offset = 0
            predictions.append(np.zeros([CAGS.H * CAGS.W], np.float32))
            for i, run in enumerate(runs):
                predictions[-1][offset:offset + run] = i % 2
                offset += run

        return CAGS.evaluate_segmentation(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--task", default="classification", type=str, help="Task to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        if args.task == "classification":
            with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
                accuracy = CAGS.evaluate_classification_file(getattr(CAGS(), args.dataset), predictions_file)
            print("CAGS accuracy: {:.2f}%".format(accuracy))

        if args.task == "segmentation":
            with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
                iou = CAGS.evaluate_segmentation_file(getattr(CAGS(), args.dataset), predictions_file)
            print("CAGS IoU: {:.2f}%".format(iou))
