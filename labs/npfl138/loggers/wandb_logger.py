# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import html
from typing import Any, Self

import torch

from .base_logger import BaseLogger
from ..metric import Metric
from ..type_aliases import AnyArray, DataFormat, TensorOrTensors


class WandBLogger(BaseLogger):
    """A W&B logger interface.

    The text values are by default also logged as HTML for better visualization.
    """
    def __init__(self, project: str, *, text_also_as_html: bool = True, **kwargs: Any) -> None:
        """Create the WandBLogger with the given project name.

        Additional keyword arguments are passed to `wandb.init()`.

        Parameters:
          project: The name of the W&B project.
          text_also_as_html: Whether to log text messages also as HTML.
            That has the advantage of interactive visualization of the value
            at different epochs and preserving whitespace formatting.
          kwargs: Additional keyword arguments passed to `wandb.init()`, for example:

            - `dir` ([`str`][str]) – The directory where the W&B files will be stored in a `wandb` subdirectory.
            - `id` ([`str`][str]) – A unique identifier for this run.
            - `name` ([`str`][str]) – A short display name for this run to use in the UI.
            - `notes` ([`str`][str]) – A detailed description of the run.
            - `tags` ([`Sequence`][collections.abc.Sequence][[`str`][str]]) – A list of tags
              to label this run in the UI.
        """
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(project=project, **kwargs)
        self._text_also_as_html = text_also_as_html

    def close(self) -> None:
        if self.run:
            self.run.finish()
            self.run = None

    def _maybe_as_html(self, label: str, text: str) -> dict[str, Any]:
        """Return a dict with the HTML version of the text if enabled.

        The text is converted to HTML-safe format and returned as a wandb.Html object.
        """
        if not self._text_also_as_html:
            return {}
        return {f"{label}_html": self.wandb.Html("<pre>" + html.escape(text) + "</pre>")}

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        audio = self.preprocess_audio(audio).numpy()
        self.run.log({label: self.wandb.Audio(audio, sample_rate=sample_rate)}, step=epoch)
        return self

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        self.run.config.update(dict(sorted(config.items())))
        config_json = self.format_config_as_json(config)
        self.run.log({"config": config_json} | self._maybe_as_html("config", config_json), step=epoch)
        return self

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        return super().log_figure(label, figure, epoch, tight_layout, close)

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        # The logging in WandB has a lot of limitations. One is that all
        # children of the graph must be executed before the graph is logged;
        # however, metrics are usually not executed during the forward pass.
        # We therefore try to exclude metrics from the logged graph.
        watched_children = torch.nn.Module()
        for name, child in graph.named_children():
            if not any(isinstance(module, Metric) for module in child.modules()):
                watched_children.add_module(name, child)
        self.run.watch(watched_children, log=None, log_graph=True)
        with self.graph_in_eval_mode(graph), torch.no_grad():
            graph(*data) if isinstance(data, (list, tuple)) else graph(data)  # Run the graph to log it.
        self.run.unwatch(watched_children)
        return self

    def log_image(self, label: str, image: AnyArray, epoch: int, data_format: DataFormat = "HWC") -> Self:
        image = self.preprocess_image(image, data_format).numpy()
        self.run.log({label: self.wandb.Image(image)}, step=epoch)
        return self

    def log_metrics(self, logs: dict[str, float], epoch: int, description: str | None = None) -> Self:
        self.run.log(logs, step=epoch)
        return self

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        self.run.log({label: text} | self._maybe_as_html(label, text), step=epoch)
        return self
