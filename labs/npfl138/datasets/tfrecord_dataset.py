# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import array
import struct
from typing import Any

import numpy as np
import torch


class TFRecordDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
        arrays, indices, self._size = self._tfrecord_load(path, size)
        if decode_on_demand:
            self._data, self._arrays, self._indices = None, arrays, indices
        else:
            self._data = [self._tfrecord_decode(arrays, indices, i) for i in range(len(self))]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Any:
        if self._data:
            return self._data[index]
        return self._tfrecord_decode(self._arrays, self._indices, index)

    def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> Any:
        raise NotImplementedError()

    @staticmethod
    def _tfrecord_load(path: str, items: int) -> tuple[dict[str, array.array], dict[str, array.array]]:
        def get_value() -> np.int64:
            nonlocal data, offset
            value, start = np.int64(data[offset] & 0x7F), offset
            offset += 1
            while data[offset - 1] & 0x80:
                value |= (data[offset] & 0x7F) << (7 * (offset - start))
                offset += 1
            return value

        def get_value_of_kind(kind: int) -> np.int64:
            nonlocal data, offset
            assert data[offset] == kind
            offset += 1
            return get_value()

        arrays, indices, size = {}, {}, 0
        with open(path, "rb") as file:
            while items < 0 or size < items:
                length = file.read(8)
                if items < 0 and not(len(length)):
                    break
                size += 1
                assert len(length) == 8
                length, = struct.unpack("<Q", length)
                assert len(file.read(4)) == 4
                data = file.read(length)
                assert len(data) == length
                assert len(file.read(4)) == 4

                offset = 0
                length = get_value_of_kind(0x0A)
                assert len(data) - offset == length
                while offset < len(data):
                    get_value_of_kind(0x0A)
                    length = get_value_of_kind(0x0A)
                    key = data[offset:offset + length].decode("utf-8")
                    offset += length
                    get_value_of_kind(0x12)
                    if key not in arrays:
                        arrays[key] = array.array({0x0A: "B", 0x1A: "q", 0x12: "f"}.get(data[offset], "B"))
                        indices[key] = array.array("Q", [0])

                    if data[offset] == 0x0A:
                        length = get_value_of_kind(0x0A) and get_value_of_kind(0x0A)
                        arrays[key].frombytes(data[offset:offset + length])
                        offset += length
                    elif data[offset] == 0x1A:
                        length = get_value_of_kind(0x1A) and get_value_of_kind(0x0A)
                        target_offset = offset + length
                        while offset < target_offset:
                            arrays[key].append(get_value())
                    elif data[offset] == 0x12:
                        length = get_value_of_kind(0x12) and get_value_of_kind(0x0A)
                        arrays[key].frombytes(np.frombuffer(
                            data, np.dtype("<f4"), length >> 2, offset).astype(np.float32).tobytes())
                        offset += length
                    else:
                        raise ValueError("Unsupported data tag {}".format(data[offset]))
                    indices[key].append(len(arrays[key]))

        typecode_to_dtype = {"B": torch.uint8, "f": torch.float32, "q": torch.int64, "Q": torch.uint64}
        for key, value in arrays.items():
            arrays[key] = torch.asarray(value or [], dtype=typecode_to_dtype[value.typecode], copy=True)
        for key, value in indices.items():
            indices[key] = torch.asarray(value or [], dtype=typecode_to_dtype[value.typecode], copy=True)
        return arrays, indices, size
