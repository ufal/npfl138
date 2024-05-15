import array
import os
import struct
import sys
from typing import Any, Callable, Sequence, TextIO, TypedDict
import urllib.request

import numpy as np
import torch
import torchvision


class Image64Dataset:
    H: int = 64
    W: int = 64
    C: int = 3

    Element = TypedDict("Element", {"image": torch.Tensor})

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, decode_on_demand: bool) -> None:
            arrays, indices = Image64Dataset._load_data(path)
            self._size = len(indices["image"]) - 1
            if decode_on_demand:
                self._data, self._arrays, self._indices = None, arrays, indices
            else:
                self._data = [self._decode(arrays, indices, i) for i in range(len(self))]

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> "Image64Dataset.Element":
            if self._data:
                return self._data[index]
            return self._decode(self._arrays, self._indices, index)

        def transform(self, transform: Callable[["Image64Dataset.Element"], Any]) -> "Image64Dataset.TransformedDataset":
            return Image64Dataset.TransformedDataset(self, transform)

        def _decode(self, data: dict, indices: dict, index: int) -> "Image64Dataset.Element":
            return {
                "image": torchvision.io.decode_image(
                    torch.frombuffer(data["image"], dtype=torch.uint8, offset=indices["image"][:-1][index],
                                     count=indices["image"][1:][index] - indices["image"][:-1][index]),
                    torchvision.io.ImageReadMode.RGB).permute(1, 2, 0),
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

    def __init__(self, name: str, decode_on_demand: bool = False) -> None:
        path = "{}.tfrecord".format(name)
        if not os.path.exists(path):
            print("Downloading file {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}.LICENSE".format(self._URL, path), filename="{}.LICENSE".format(path))
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        self.train = self.Dataset(path, decode_on_demand)

    train: Dataset

    # TFRecord loading
    @staticmethod
    def _load_data(path: str) -> tuple[dict[str, array.array], dict[str, array.array]]:
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
            while len(length := file.read(8)):
                assert len(length) == 8
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
