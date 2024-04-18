import array
import os
import struct
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch
import torchaudio
import torchmetrics


# A class for managing mapping between strings and indices.
# It provides:
# - `__len__`: number of strings in the vocabulary
# - `__iter__`: iterator over strings in the vocabulary
# - `string(index: int) -> str`: string for a given index to the vocabulary
# - `strings(indices: Sequence[int]) -> list[str]`: list of strings for given indices
# - `index(string: str) -> int`: index of a given string in the vocabulary
# - `indices(strings: Sequence[str]) -> list[int]`: list of indices for given strings
class Vocabulary:
    def __init__(self, strings: Sequence[str]) -> None:
        self._strings = list(strings)
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
        return self._string_map[string]

    def indices(self, strings: Sequence[str]) -> list[int]:
        return [self._string_map[string] for string in strings]


class CommonVoiceCs:
    MFCC_DIM: int = 13

    LETTERS: list[str] = [
        " ", "a", "á", "ä", "b", "c", "č", "d", "ď", "e", "é", "è", "ě",
        "f", "g", "h", "i", "í", "ï", "j", "k", "l", "m", "n", "ň", "o",
        "ó", "ö", "p", "q", "r", "ř", "s", "š", "t", "ť", "u", "ú", "ů",
        "ü", "v", "w", "x", "y", "ý", "z", "ž",
    ]

    Element = TypedDict("Element", {"mfccs": torch.Tensor, "sentence": str})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            self._size = size

            arrays, indices = CommonVoiceCs._load_data(path, size)
            if decode_on_demand:
                self._data, self._arrays, self._indices = None, arrays, indices
            else:
                self._data = [self._decode(arrays, indices, i) for i in range(size)]

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> "CommonVoiceCs.Element":
            if self._data:
                return self._data[index]
            return self._decode(self._arrays, self._indices, index)

        def transform(self, transform: Callable[["CommonVoiceCs.Element"], Any]) -> "CommonVoiceCs.TransformedDataset":
            return CommonVoiceCs.TransformedDataset(self, transform)

        def _decode(self, data: dict, indices: dict, index: int) -> "CommonVoiceCs.Element":
            return {
                "mfccs": torch.frombuffer(
                    data["mfccs"], dtype=torch.float32, offset=indices["mfccs"][:-1][index],
                    count=indices["mfccs"][1:][index] - indices["mfccs"][:-1][index]).view(-1, CommonVoiceCs.MFCC_DIM),
                "sentence": data["sentence"][
                    indices["sentence"][index]:indices["sentence"][index + 1]].tobytes().decode("utf-8"),
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

        def transform(self, transform: Callable[..., Any]) -> "CommonVoiceCs.TransformedDataset":
            return CommonVoiceCs.TransformedDataset(self, transform)

    def __init__(self, decode_on_demand: bool = False) -> None:
        for dataset, size in [("train", 9_773), ("dev", 904), ("test", 3_240)]:
            path = "common_voice_cs.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset, self.Dataset(path, size, decode_on_demand))

        self._letters_vocab = Vocabulary(self.LETTERS)

    train: Dataset
    dev: Dataset
    test: Dataset

    @property
    def letters_vocab(self) -> Vocabulary:
        return self._letters_vocab

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
                        arrays[key] = array.array({0x0A: "B", 0x1A: "q", 0x12: "f"}.get(data[offset], "B"))
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

    # Methods for generating MFCCs.
    def load_audio(self, path: str, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
        audio, sample_rate = torchaudio.load(path)
        if target_sample_rate is not None and target_sample_rate != sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        return torch.mean(audio, dim=0), sample_rate

    # Note that while the dataset MFCCs were generated using an implementation
    # functionally equivalent to the following, different resampling was used,
    # so the values are not exactly the same.
    def mfcc_extract(self, audio: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        assert sample_rate == 16000, "Only 16k sample rate is supported"

        if not hasattr(self, "_mfcc_fn"):
            # Compute a 1024-point STFT with frames of 64 ms and 75% overlap.
            # Then warp the linear scale spectrograms into the mel-scale.
            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            # Finally, compute MFCCs from log-mel-spectrograms and take the first
            # `CommonVoiceCs.MFCC_DIM=13` of them.
            self._mfcc_fn = torchaudio.transforms.MFCC(
                sample_rate=16_000, n_mfcc=self.MFCC_DIM, log_mels=True,
                melkwargs={"n_fft":1024, "win_length":1024, "hop_length":256,
                           "f_min": 80., "f_max": 7600., "n_mels": 80, "center": False}
            )
        # Compute MFCCs of shape `[sequence_length, CommonVoiceCs.MFCC_DIM=13]`.
        mfccs = self._mfcc_fn(audio).permute(1, 0)
        mfccs[:, 0] *= 2**0.5  # Scale the first coefficient for consistency with the dataset
        return mfccs

    # Torchmetric for computing mean edit distance
    class EditDistanceMetric(torchmetrics.MeanMetric):
        def update(self, pred: Sequence[Sequence[Any]], true: Sequence[Sequence[Any]]) -> None:
            edit_distances = []
            for y_pred, y_true in zip(pred, true):
                edit_distances.append(torchaudio.functional.edit_distance(y_pred, y_true) / len(y_true))
            return super().update(edit_distances)

    # Evaluation infrastructure
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[str]) -> float:
        gold = [example["sentence"] for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        edit_distance = CommonVoiceCs.EditDistanceMetric()
        for gold_sentence, prediction in zip(gold, predictions):
            edit_distance([prediction], [gold_sentence])
        return 100 * edit_distance.compute()

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line in predictions_file:
            predictions.append(line.rstrip("\n"))
        return CommonVoiceCs.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            edit_distance = CommonVoiceCs.evaluate_file(getattr(CommonVoiceCs(), args.dataset), predictions_file)
        print("CommonVoiceCs edit distance: {:.2f}%".format(edit_distance))
