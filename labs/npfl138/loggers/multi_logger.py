# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Iterable
from typing import Any, Self

import torch

from .base_logger import BaseLogger
from ..logger import Logger
from ..type_aliases import AnyArray, DataFormat, TensorOrTensors


class MultiLogger(BaseLogger):
    """A logger that forwards all logging calls to multiple other loggers."""

    def __init__(self, loggers: Iterable[Logger]) -> None:
        """Initialize the multi-logger.

        Parameters:
          loggers: An iterable of loggers to forward the logging calls to.
        """
        self.loggers: list[Logger] = list(loggers)

    loggers: list[Logger]
    """The list of loggers to forward the logging calls to."""

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        for logger in self.loggers:
            logger.log_audio(label, audio, sample_rate, epoch)
        return self

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        for logger in self.loggers:
            logger.log_config(config, epoch)
        return self

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        import matplotlib.pyplot as plt
        for i, logger in enumerate(self.loggers):
            logger.log_figure(label, figure, epoch, tight_layout, False)
        close and plt.close(figure)
        return self

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        for logger in self.loggers:
            logger.log_graph(graph, data, epoch)
        return self

    def log_image(self, label: str, image: AnyArray, epoch: int, data_format: DataFormat = "HWC") -> Self:
        for logger in self.loggers:
            logger.log_image(label, image, epoch, data_format)
        return self

    def log_metrics(self, logs: dict[str, float], epoch: int, description: str | None = None) -> Self:
        for logger in self.loggers:
            logger.log_metrics(logs, epoch, description)
        return self

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        for logger in self.loggers:
            logger.log_text(label, text, epoch)
        return self
