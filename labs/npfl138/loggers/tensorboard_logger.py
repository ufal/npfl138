# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os
from typing import Any, Self

import torch
import torch.utils.tensorboard

from .base_logger import BaseLogger
from ..type_aliases import AnyArray, DataFormat, TensorOrTensors


class TensorBoardLogger(BaseLogger):
    """A TensorBoard logger interface.

    In addition to implementing the [Logger][npfl138.Logger] interface, it provides
    a method for obtaining the underlying TensorBoard
    [SummaryWriter][torch.utils.tensorboard.writer.SummaryWriter] instance for a given
    writer name.
    """
    def __init__(self, logdir: str) -> None:
        """Initialize the TensorBoard logger.

        Parameters:
          logdir: The root directory where the TensorBoard logs will be stored.
        """
        self._logdir: str = logdir
        self._writers: dict[str, torch.utils.tensorboard.writer.SummaryWriter] = {}

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def get_writer(self, name: str) -> torch.utils.tensorboard.writer.SummaryWriter:
        """Possibly create and return a TensorBoard writer for the given name.

        Returns:
          writer: The opened TensorBoard writer.
        """
        if name not in self._writers:
            self._writers[name] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._logdir, name))
        return self._writers[name]

    def _get_writer_from_label(self, label: str) -> tuple[torch.utils.tensorboard.writer.SummaryWriter, str]:
        """Possibly create and return a TensorBoard writer for the given label.

        Returns:
          writer: The opened TensorBoard writer.
          label: The label without the writer prefix.
        """
        writer, label = label.split(":", maxsplit=1) if ":" in label else ("train", label)
        return self.get_writer(writer), label

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        audio = self.preprocess_audio(audio)
        audio = (audio.to(torch.float32) / 32_767).clamp(-1.0, 1.0).movedim(-1, 0)
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        writer, label = self._get_writer_from_label(label)
        writer.add_audio(label, audio, epoch, sample_rate=sample_rate)
        writer.flush()
        return self

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        writer = self.get_writer("train")
        writer.add_text("config", self.format_config_as_json(config), epoch)
        writer.flush()
        return self

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        return super().log_figure(label, figure, epoch, tight_layout, close)

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        writer = self.get_writer("train")
        writer.add_graph(graph, data, use_strict_trace=False)
        writer.flush()
        return self

    def log_image(self, label: str, image: AnyArray, epoch: int, data_format: DataFormat = "HWC") -> Self:
        image = self.preprocess_image(image, data_format)
        writer, label = self._get_writer_from_label(label)
        writer.add_image(label, image, epoch, dataformats="HWC" if image.ndim == 3 else "HW")
        writer.flush()
        return self

    def log_metrics(self, logs: dict[str, float], epoch: int, description: str | None = None) -> Self:
        for label, value in logs.items():
            writer, label = self._get_writer_from_label(label)
            writer.add_scalar(label, value, epoch)

        for writer in self._writers.values():
            writer.flush()

        return self

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        writer, label = self._get_writer_from_label(label)
        writer.add_text(label, text, epoch)
        writer.flush()
        return self
