# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from typing import Any, Self

import torch

from .type_aliases import AnyArray, DataFormat, TensorOrTensors


class Logger(ABC):
    """An abstract logger interface for logging metrics and other information."""

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release its resources."""
        ...

    @abstractmethod
    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        """Log the given audio with the given label at the given epoch.

        Parameters:
          label: The label of the logged audio.
          audio: The audio to log, represented as an array with any of the
            following shapes:

            - `(L,)` of `(L, 1)` for mono audio,
            - `(L, 2)` for stereo audio.

            If the sample values are floating-point numbers, they are expected
            to be in the `[-1, 1]` range; otherwise, they are assumed to be in the
            `[-32_768, 32_767]` range.
          sample_rate: The sample rate of the audio.
          epoch: The epoch number at which the audio is logged.
        """
        ...

    @abstractmethod
    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        """Log the given configuration dictionary at the given epoch.

        Parameters:
          config: A JSON-serializable dictionary representing the configuration to log.
          epoch: The epoch number at which the configuration is logged.
        """
        ...

    @abstractmethod
    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        """Log the given matplotlib Figure with the given label at the given epoch.

        Parameters:
          label: The label of the logged image.
          figure: A matplotlib Figure.
          epoch: The epoch number at which the image is logged.
          tight_layout: Whether to apply tight layout to the figure before logging it.
          close: Whether to close the figure after logging it.
        """
        ...

    @abstractmethod
    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        """Log the given computation graph by tracing it with the given data.

        Alternatively, loggers may choose to log the graph using TorchScript, run
        it on the given data, or use any other mechanism they see fit.

        Parameters:
          graph: The computation graph to log, represented as a PyTorch module.
          data: The input data to use for tracing the computation graph.
          epoch: The epoch number at which the computation graph is logged.
        """
        ...

    @abstractmethod
    def log_image(self, label: str, image: AnyArray, epoch: int, data_format: DataFormat = "HWC") -> Self:
        """Log the given image with the given label at the given epoch.

        Parameters:
          label: The label of the logged image.
          image: The image to log, represented as a `PIL` image or as an array
            of any of the following shapes (assuming `"HWC"` data format):

            - `(H, W)` or `(H, W, 1)` for grayscale images,
            - `(H, W, 2)` for grayscale images with alpha channel,
            - `(H, W, 3)` for RGB images,
            - `(H, W, 4)` for RGBA images.

            If the pixel values are floating-point numbers, they are expected
            to be in the `[0, 1]` range; otherwise, they are assumed to be in the
            `[0, 255]` range.
          epoch: The epoch number at which the image is logged.
          data_format: The data format of the image specifying whether the channels
            are stored in the last dimension (`"HWC"`, the default) or in the first dimension (`"CHW"`);
            ignored for a `PIL` image.
        """
        ...

    @abstractmethod
    def log_metrics(self, logs: dict[str, float], epoch: int, description: str | None = None) -> Self:
        """Log metrics collected during a given epoch, with an optional description.

        Parameters:
          logs: A dictionary of logged metrics for the epoch.
          epoch: The epoch number at which the logs were collected.
          description: An optional description of the logged metrics (used only by some loggers).
        """
        ...

    @abstractmethod
    def log_text(self, label: str, text: str, epoch: int) -> Self:
        """Log the given text with the given label at the given epoch.

        Parameters:
          label: The label of the logged text.
          text: The text to log.
          epoch: The epoch number at which the text is logged.
        """
        ...
