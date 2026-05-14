# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Image64Dataset is a dataset of unlabeled color 64x64 images."""
from typing import TypedDict

import torch
import torchvision

from .downloader import download_url_to_file
from .tfrecord_dataset import TFRecordDataset


class Image64Dataset:
    C: int = 3
    """The number of channels of the images."""
    H: int = 64
    """The height of the images."""
    W: int = 64
    """The width of the images."""

    Element = TypedDict("Element", {"image": torch.Tensor})
    """The type of a single dataset element."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets"

    class Dataset(TFRecordDataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            super().__init__(path, size, decode_on_demand)

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return super().__len__()

        def __getitem__(self, index: int) -> "Image64Dataset.Element":
            """Return the `index`-th element of the dataset."""
            return super().__getitem__(index)

        def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> "Image64Dataset.Element":
            return {
                "image": torchvision.io.decode_image(
                    data["image"][indices["image"][index]:indices["image"][index + 1]],
                    torchvision.io.ImageReadMode.RGB),
            }

    def __init__(self, name: str, decode_on_demand: bool = False) -> None:
        """Load the given dataset, downloading it if necessary.

        Parameters:
          name: The name of the dataset, for example `oxford_flowers102`.
          decode_on_demand: if `False` (the default), the images are fully
            decoded when the dataset is loaded; if `True`, the images are
            loaded as byte strings and decoded on demand when accessed.
        """
        path = download_url_to_file(self.URL, f"{name}.tfrecord", f"{name}.tfrecord.LICENSE")
        self.train = self.Dataset(path, -1, decode_on_demand)

    train: Dataset
    """The training dataset."""
