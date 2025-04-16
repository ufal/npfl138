# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The Keras-inspired high-level API for training PyTorch models.

The [TrainableModule][npfl138.trainable_module.TrainableModule] class is
a high-level API for training PyTorch models. It is a subclass of
[torch.nn.Module][] and:

- It provides a high-level API for training, evaluation, and prediction
  via [fit][npfl138.TrainableModule.fit], [evaluate][npfl138.TrainableModule.evaluate],
  and [predict][npfl138.TrainableModule.predict] methods. Each can be customized
  by overriding the corresponding [train_step][npfl138.TrainableModule.train_step],
  [test_step][npfl138.TrainableModule.test_step], or [predict_step][npfl138.TrainableModule.predict_step]
  methods.

- The module automatically handles moving the model to a specified device,
  using the first available accelerator (GPU, MPS, XPU) by default. To this end,
  [configure][npfl138.TrainableModule.configure] or
  [load_weights][npfl138.TrainableModule.load_weights] must always be called
  first before using the high-level API.

- The module provides API for serialization and deserialization of the model,
  both the weights ([save_weights][npfl138.TrainableModule.save_weights],
  [load_weights][npfl138.TrainableModule.load_weights]) and the configuration
  ([save_config][npfl138.TrainableModule.save_config],
  [load_config][npfl138.TrainableModule.load_config]).

- The module keeps a collection of metrics implementing the
  [MetricProtocol][npfl138.trainable_module.MetricProtocol] (e.g., any
  metric from `torchmetric`), and stores the computed logs in a text
  file, in TensorBoard logs, and in the console.
"""

import argparse
import json
import os
import sys
import time
from typing import Literal, Protocol, TextIO, TypeAlias, TypeVar

import numpy as np
import torch
import torch.utils.tensorboard
import tqdm

Tensor: TypeAlias = torch.Tensor | torch.nn.utils.rnn.PackedSequence
"""A type alias for a single tensor or a packed sequence of tensors."""
TensorOrTensors: TypeAlias = Tensor | tuple[Tensor, ...] | list[Tensor]
"""A type alias for a single tensor/packed sequence of a sequence of them."""

Logs: TypeAlias = dict[str, float]
"""A dictionary of logs, with keys being the log names and values being the log values."""

Self: TypeVar = TypeVar("Self", bound="TrainableModule")


class LossProtocol(Protocol):
    def __call__(self, y_pred: TensorOrTensors, y: TensorOrTensors) -> torch.Tensor:
        """Compute the loss of the given predictions and gold outputs."""
        ...


class MetricProtocol(Protocol):
    def reset(self) -> None:
        """Reset the metric to its initial state."""
        ...
    def update(self, y_pred: TensorOrTensors, y: TensorOrTensors) -> None:  # noqa: E301
        """Update the metric with the given predictions and gold outputs."""
        ...
    def compute(self) -> torch.Tensor:  # noqa: E301
        """Return the current value of the metric."""
        ...


class CallbackProtocol(Protocol):
    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> Literal["stop_training"] | None:
        """Represents a callback called after every training epoch.

        If the callback returns [TrainableModule.STOP_TRAINING][npfl138.TrainableModule.STOP_TRAINING],
        the training stops.

        Parameters:
          module: the module being trained
          epoch: the current epoch number (one-based)
          logs: a dictionary of logs, newly computed metric or losses should be added here

        Returns:
          [TrainableModule.STOP_TRAINING][npfl138.TrainableModule.STOP_TRAINING] to stop the training;
          `None` to continue.
        """
        ...


class KeepPrevious:
    pass
keep_previous = KeepPrevious()  # noqa: E305


def is_sequence(x: TensorOrTensors) -> bool:
    """Check if the given object is a sequence of tensors.

    The method is used in the TrainableModule to distinguish between tensors
    and tuples. However, torch.nn.utils.rnn.PackedSequence is technically
    a NamedTuple and therefore fulfills the `isinstance(x, tuple)` condition.
    """
    return isinstance(x, list) or (isinstance(x, tuple) and not isinstance(x, torch.nn.utils.rnn.PackedSequence))


def maybe_unpack(x: Tensor, as_numpy: bool) -> Tensor | np.ndarray | list[Tensor] | list[np.ndarray]:
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        return [y.numpy(force=True) if as_numpy else y for y in torch.nn.utils.rnn.unpack_sequence(x)]
    return x.numpy(force=True) if as_numpy else x


def check_tensor(x: Tensor) -> bool:
    return isinstance(x, (torch.Tensor, torch.nn.utils.rnn.PackedSequence))


def check_tensors(x: TensorOrTensors) -> bool:
    return check_tensor(x) if not is_sequence(x) else all(map(check_tensor, x))


def validate_batch_input_output(
    batch: tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors],
) -> tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors]:
    assert isinstance(batch, (tuple, list)) and len(batch) == 2, "The batch must be an (input, output) pair."
    assert check_tensors(batch[0]), "The batch input must be a tensor or a tuple of tensors."
    assert check_tensors(batch[1]), "The batch output must be a tensor or a tuple of tensors."
    return batch


def validate_batch_input(
    batch: TensorOrTensors | tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors], with_labels: bool,
) -> TensorOrTensors:
    if with_labels:
        assert isinstance(batch, (tuple, list)) and len(batch) == 2, "The batch must be an (input, _) pair."
        batch = batch[0]
    assert check_tensors(batch), "The batch input must be a tensor or a tuple of tensors."
    return batch


def console_default(default: int) -> int:
    try:
        return int(os.environ["CONSOLE"])
    except Exception:
        return default


def get_auto_device() -> torch.device:
    """Return the first available accelerator or CPU if none is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class LossTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64), persistent=False)

    def reset(self):
        self.total.zero_()
        self.count.zero_()

    def __call__(self, value):
        total, count = self.total, self.count
        total.add_(value)
        count.add_(1)
        return total / count


class TrainableModule(torch.nn.Module):
    """A simple Keras-like module for training with raw PyTorch.

    The module provides fit/evaluate/predict methods, computes loss and metrics,
    and generates TensorBoard, text file, and console logs. By default, it uses
    an accelerator (GPU, MPS, XPU) if available, and CPU otherwise.

    The input to the model can be either a single tensor/PackedSequence or
    a tuple of those. Similarly, the output can be a single tensor/PackedSequence
    or a tuple of those. However, when there are multiple outputs, you
    must handle loss and metrics computation manually.
    """
    STOP_TRAINING: Literal["stop_training"] = "stop_training"
    """A constant returned by callbacks to stop the training."""

    def __init__(self, module: torch.nn.Module | None = None):
        """Initialize the module, optionally with an existing PyTorch module.

        Parameters:
          module: An optional existing PyTorch module to wrap, e.g., a [torch.nn.Sequential][]
            or a pretrained Transformer. If given, the module still must be configured.
        """
        super().__init__()
        self.device = None
        self.unconfigure()
        if module is not None:
            self.module = module
            self.forward = self._call_wrapped_module

    def _call_wrapped_module(self, inputs):
        return self.module(inputs)

    def configure(
        self,
        *,
        optimizer: torch.optim.Optimizer | None | KeepPrevious = keep_previous,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None | KeepPrevious = keep_previous,
        loss: LossProtocol | None | KeepPrevious = keep_previous,
        metrics: dict[str, MetricProtocol] | KeepPrevious = keep_previous,
        initial_epoch: int | KeepPrevious = keep_previous,
        logdir: str | None | KeepPrevious = keep_previous,
        device: torch.device | str | Literal["auto"] | KeepPrevious = keep_previous,
    ) -> Self:
        """Configure the module fitting, evaluation, and placement.

        The method can be called multiple times, preserving previously set values by default.

        Note:
          When an input argument cannot be `None`, the corresponding field is
          never `None` after this call.

        Parameters:
          optimizer: The optimizer to use for training.
          scheduler: An optional learning rate scheduler used after every batch.
          loss: The loss function to minimize, implementing the
            [LossProtocol][npfl138.trainable_module.LossProtocol].
          metrics: A dictionary of additional metrics to compute, each being an
            object implementing the [MetricProtocol][npfl138.trainable_module.MetricProtocol]
            (reset/update/compute), e.g., a `torchmetrics.Metric`.
          initial_epoch: The initial epoch of the model used during training and evaluation.
          logdir: An optional directory where textual and TensorBoard logs should be stored.
          device: The device to move the module to. When "auto", or `keep_previous`
            with no previously set device, the first of cuda/mps/xpu is used if available.

        Returns:
          self
        """
        self.optimizer = optimizer if optimizer is not keep_previous else self.optimizer
        self.scheduler = scheduler if scheduler is not keep_previous else self.scheduler
        self.loss = loss if loss is not keep_previous else self.loss
        self.loss_tracker = self.loss_tracker or LossTracker()
        if metrics is not keep_previous or not self.metrics:
            self.metrics = torch.nn.ModuleDict({} if metrics is keep_previous else metrics)
        self.epoch = initial_epoch if initial_epoch is not keep_previous else self.epoch or 0
        if logdir is not keep_previous and logdir != self.logdir:  # reset loggers on a new logdir
            self._log_file, self._tb_writers = None, {}
        self.logdir = logdir if logdir is not keep_previous else self.logdir
        if device is not keep_previous or not self.device:
            self.device = get_auto_device() if device == "auto" or device is keep_previous else torch.device(device)
        self.to(self.device)
        return self

    def unconfigure(self) -> Self:
        """Remove all training configuration of the TrainableModule.

        Only the module device is kept.

        Returns:
          self
        """
        self.optimizer, self.scheduler, self.epoch = None, None, None
        self.loss, self.loss_tracker, self.metrics = None, None, None
        self.logdir, self._log_file, self._tb_writers = None, None, None
        return self

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        epochs: int,
        dev: torch.utils.data.DataLoader | None = None,
        callbacks: list[CallbackProtocol] = [],
        log_graph: bool = False,
        console: int = console_default(2),
    ) -> Logs:
        """Train the model on the given dataset.

        Parameters:
          dataloader: The training dataset, each element a pair of inputs and outputs;
            the inputs and outputs can be either single tensors or sequences of tensors.
          epochs: The number of epochs to train.
          dev: An optional development dataset to evaluate after every epoch, with the
            same format as the training dataset.
          callbacks: A list of callbacks to call after every epoch, each implementing
            the [CallbackProtocol][npfl138.trainable_module.CallbackProtocol]
            with arguments `self`, `epoch`, and `logs`, possibly returning
            [TrainableModule.STOP_TRAINING](npfl138.TrainableModule.STOP_TRAINING] to stop the
            training (note that the module is set to evaluation mode before calling each callback).
          log_graph: Controls whether to log the model graph to TensorBoard.
          console: Controls the console verbosity: 0 for silent, 1 for epoch logs, 2 for
            additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `CONSOLE` environment variable.

        Returns:
          logs: A dictionary of logs from the training and optionally dev evaluation.

        Note:
          The module is set to evaluation mode when returning from this method.
        """
        assert self.loss_tracker is not None, "The TrainableModule has not been configured, run configure first."
        logs, epochs, stop_training = {}, self.epoch + epochs, False
        while self.epoch < epochs and not stop_training:
            self.epoch += 1
            self.train()
            self.loss_tracker.reset()
            for metric in self.metrics.values():
                metric.reset()
            start = time.time()
            epoch_message = f"Epoch {self.epoch}/{epochs}"
            data_and_progress = tqdm.tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if console == 2 else console < 2)
            for batch in data_and_progress:
                xs, y = validate_batch_input_output(batch)
                xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
                y = tuple(y_.to(self.device) for y_ in y) if is_sequence(y) else y.to(self.device)
                log_graph = log_graph and self.log_graph(xs) and False
                logs = self.train_step(xs, y)
                if not data_and_progress.disable:
                    logs_message = " ".join([f"{k}={v:#.{0<abs(v)<2e-4 and '2e' or '4f'}}" for k, v in logs.items()])
                    data_and_progress.set_description(f"{epoch_message} {logs_message}", refresh=False)
            logs = {f"train_{k}": v for k, v in logs.items()}
            if dev is not None:
                logs |= {f"dev_{k}": v for k, v in self.eval().evaluate(dev, log_as=None).items()}
            for callback in callbacks:
                stop_training = callback(self.eval(), self.epoch, logs) == self.STOP_TRAINING or stop_training
            self.log_metrics(logs, epochs, time.time() - start, console)
        self.eval()
        return logs

    def train_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single training step, returning the logs.

        Parameters:
          xs: The input batch to the model, either a single tensor or a sequence of tensors.
          y: The target output batch of the model, either a single tensor or a sequence of tensors.

        Returns:
          logs: A dictionary of logs from the training step.
        """
        self.optimizer.zero_grad()
        y_pred = self(*xs)
        loss = self.compute_loss(y_pred, y, *xs)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.scheduler is not None and self.scheduler.step()
            return {"loss": self.loss_tracker(loss)} \
                | ({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {}) \
                | self.compute_metrics(y_pred, y, *xs)

    def compute_loss(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: tuple[Tensor]) -> torch.Tensor:
        """Compute the loss of the model given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a sequence of tensors.
          y: The target output of the model, either a single tensor or a sequence of tensors.
          *xs: The inputs to the model, unpacked, if the input was a sequence of tensors.

        Returns:
          loss: The computed loss.
        """
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors) -> Logs:
        """Compute and return metrics given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a sequence of tensors.
          y: The target output of the model, either a single tensor or a sequence of tensors.
          *xs: The inputs to the model, unpacked, if the input was a sequence of tensors.

        Returns:
          logs: A dictionary of computed metrics.
        """
        for metric in self.metrics.values():
            metric.update(y_pred, y)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        log_as: str | None = "test",
        callbacks: list[CallbackProtocol] = [],
        console: int = console_default(1),
    ) -> Logs:
        """An evaluation of the model on the given dataset.

        Parameters:
          dataloader: The dataset to evaluate on, each element a pair of inputs and outputs;
            the inputs and outputs can be either a single tensor or a sequence of tensors.
          log_as: The name of the dataset used in the logs; when `None`, no logs are written.
          callbacks: A list of callbacks to call after the evaluation, each implementing
            the [CallbackProtocol][npfl138.trainable_module.CallbackProtocol] with arguments
            `self`, `epoch`, and `logs` arguments.
          console: Controls the console verbosity: 0 for silent, 1 for a single message.
            The default is 1, but can be overridden by the `CONSOLE` environment variable.
        """
        assert self.loss_tracker is not None, "The TrainableModule has not been configured, run configure first."
        self.eval()
        self.loss_tracker.reset()
        for metric in self.metrics.values():
            metric.reset()
        start = time.time()
        for batch in dataloader:
            xs, y = validate_batch_input_output(batch)
            xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
            y = tuple(y_.to(self.device) for y_ in y) if is_sequence(y) else y.to(self.device)
            logs = self.test_step(xs, y)
        if log_as is not None:
            logs = {f"{log_as}_{k}": v for k, v in logs.items()}
        for callback in callbacks:
            callback(self, self.epoch, logs)
        if log_as is not None:
            self.log_metrics(logs, elapsed=time.time() - start, console=console)
        return logs

    def test_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single evaluation step, returning the logs.

        Parameters:
        xs: The input batch to the model, either a single tensor or a sequence of tensors.
        y: The target output batch of the model, either a single tensor or a sequence of tensors.

        Returns:
          logs: A dictionary of logs from the evaluation step.
        """
        with torch.no_grad():
            y_pred = self(*xs)
            loss = self.compute_loss(y_pred, y, *xs)
            return {"loss": self.loss_tracker(loss)} | self.compute_metrics(y_pred, y, *xs)

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        data_with_labels: bool = False,
        as_numpy: bool = True,
    ) -> list[torch.Tensor | tuple[torch.Tensor, ...] | np.ndarray | tuple[np.ndarray, ...]]:
        """Compute predictions for the given dataset.

        Parameters:
          dataloader: The dataset to predict on, each element either directly the input
            or a pair whose first element is the input; the input can be either
            a single tensor or a sequence of tensors.
          data_with_labels: Specifies whether the dataloader elements
            are (input, labels) pairs or just inputs (the default).
          as_numpy: A a flag controlling whether the output should be
            converted to a numpy array or kept as a PyTorch tensor.

        Returns:
          predictions: A Python list whose elements are predictions
            of the individual examples. Note that if the input was padded, so
            will be the predictions, which will then need to be trimmed.
        """
        assert self.device is not None, "No device has been set for the TrainableModule, run configure first."
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = validate_batch_input(batch, with_labels=data_with_labels)
            xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
            y = self.predict_step(xs, as_numpy=as_numpy)
            predictions.extend(y if not isinstance(y, tuple) else zip(*y))
        return predictions

    def predict_step(
        self, xs: TensorOrTensors, as_numpy: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | np.ndarray | tuple[np.ndarray, ...]:
        """An overridable method performing a single prediction step.

        Parameters:
          xs: The input batch to the model, either a single tensor or a sequence of tensors.
          as_numpy: A flag controlling whether the output should be converted to a numpy array.

        Returns:
          predictions: The batch prediction.
        """
        with torch.no_grad():
            y = self(*xs)
            return maybe_unpack(y, as_numpy) if not is_sequence(y) else tuple(maybe_unpack(y_, as_numpy) for y_ in y)

    def save_weights(self, path: str, optimizer_path: str | None = None) -> Self:
        """Save the model weights to the given path.

        Parameters:
          path: The path to save the model weights to; a `.pt` extension is recommended.
          optimizer_path: An optional path to save the optimizer state to, relative to the
            model weights path.

        Returns:
          self
        """
        state_dict = self.state_dict()
        os.path.dirname(path) and os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

        # Save the number of epochs, optimizer state, and the scheduler state when requested.
        if optimizer_path is not None:
            optimizer_state = {"epoch": self.epoch}
            self.optimizer is not None and optimizer_state.update(optimizer=self.optimizer.state_dict())
            self.scheduler is not None and optimizer_state.update(scheduler=self.scheduler.state_dict())
            optimizer_path = os.path.join(os.path.dirname(path), optimizer_path)
            os.path.dirname(optimizer_path) and os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            torch.save(optimizer_state, optimizer_path)
        return self

    def load_weights(self, path: str, optimizer_path: str | None = None,
                     device: torch.device | str | Literal["auto"] | KeepPrevious = keep_previous) -> Self:
        """Load the model weights from the given path.

        Parameters:
          path: The path to load the model weights from.
          optimizer_path: An optional path to load the optimizer state from, relative to the
            model weights path.
          device: The device to load the model to; when "auto", or `keep_previous` with no previously
            set device, the first of cuda/mps/xpu is used if available.

        Returns:
          self
        """
        if device is not keep_previous or not self.device:
            self.device = get_auto_device() if device == "auto" or device is keep_previous else torch.device(device)
        self.load_state_dict(torch.load(path, map_location=self.device))

        # Load the number of epochs, optimizer state, and the scheduler state when requested.
        if optimizer_path is not None:
            optimizer_path = os.path.join(os.path.dirname(path), optimizer_path)
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.epoch = optimizer_state["epoch"]
            if self.optimizer is not None:
                assert "optimizer" in optimizer_state, "The optimizer state is missing."
                self.optimizer.load_state_dict(optimizer_state["optimizer"])
            else:
                assert "optimizer" not in optimizer_state, "The optimizer state is present, but there is no optimizer."
            if self.scheduler is not None:
                assert "scheduler" in optimizer_state, "The scheduler state is missing."
                self.scheduler.load_state_dict(optimizer_state["scheduler"])
            else:
                assert "scheduler" not in optimizer_state, "The scheduler state is present, but there is no scheduler."
        self.to(self.device)
        return self

    @staticmethod
    def save_config(path: str, config: dict = {}, /, **kwargs: dict) -> None:
        """Save a JSON-serializable configuration to the given path.

        The configuration can be given as a dictionary or as keyword arguments
        and the configuration values might also be [argparse.Namespace][] objects.

        Parameters:
          path: The path to save the configuration to; a `.json` extension is recommended.
          config: The configuration dictionary to save.
          **kwargs: Additional configuration values to save.
        """
        config = dict((k + " : argparse.Namespace", vars(v)) if isinstance(v, argparse.Namespace) else (k, v)
                      for k, v in {**config, **kwargs}.items())
        os.path.dirname(path) and os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_config(path: str) -> dict:
        """Load a JSON-serializable configuration from the given path.

        Parameters:
          path: The path to load the configuration from.

        Returns:
          config: The loaded configuration dictionary.
        """
        with open(path, "r", encoding="utf-8-sig") as config_file:
            config = json.load(config_file)
        return dict((k.removesuffix(" : argparse.Namespace"), argparse.Namespace(**v))
                    if k.endswith(" : argparse.Namespace") else (k, v) for k, v in config.items())

    def log_metrics(
        self, logs: Logs, epochs: int | None = None, elapsed: float | None = None, console: int = console_default(1),
    ) -> Self:
        """Log the given dictionary to file logs, TensorBoard logs, and optionally the console.

        Parameters:
          logs: The dictionary of logs to write.
          epochs: An optional total number of epochs, used during logging the epoch number.
          elapsed: An optional time elapsed since the beginning of the current epoch.
          console: Controls the console verbosity: 0 for silent, 1 for epoch logs.
            The default is 1, but can be overridden by the `CONSOLE` environment variable.

        Returns:
          self
        """
        if self.logdir is not None:
            writers = {}
            for key, value in logs.items():
                writer, metric = key.split("_", maxsplit=1) if "_" in key else ("train", key)
                writers.setdefault(writer, self.get_tb_writer(writer)).add_scalar(metric, value, self.epoch)
            for writer in writers.values():
                writer.flush()
        for file in ([self.get_log_file()] if self.logdir is not None else []) + [sys.stdout] * bool(console):
            print(f"Epoch {self.epoch}" + (f"/{epochs}" if epochs is not None else ""),
                  *[f"{elapsed:.1f}s"] if elapsed is not None else [],
                  *[f"{k}={v:#.{0<abs(v)<2e-4 and '2e' or '4f'}}" for k, v in logs.items()], file=file, flush=True)
        return self

    def log_config(self, config: dict, sort_keys: bool = True, console: int = console_default(1)) -> Self:
        """Log the given dictionary to the file logs, TensorBoard logs, and optionally the console.

        Parameters:
          config: The dictionary of configuration to write.
          sort_keys: Whether to sort the keys of the configuration dictionary.
          console: Controls the console verbosity: 0 for silent, 1 for epoch logs.
            The default is 1, but can be overridden by the `CONSOLE` environment variable.

        Returns:
          self
        """
        if self.logdir is not None:
            config = dict(sorted(config.items())) if sort_keys else config
            writer = self.get_tb_writer("train")
            writer.add_text("config", json.dumps(config, ensure_ascii=False, indent=2), self.epoch)
            writer.flush()
        for file in ([self.get_log_file()] if self.logdir is not None else []) + [sys.stdout] * bool(console):
            print("Config", f"epoch={self.epoch}", *[f"{k}={v}" for k, v in config.items()], file=file, flush=True)
        return self

    def log_graph(self, data: torch.utils.data.DataLoader | TensorOrTensors, data_with_labels: bool = False) -> Self:
        """Log the traced module as a graph to the TensorBoard logs.

        Tracing requires an example batch; either the first batch from the
        dataloader passed in `data` is used, or the `data` itself is used.

        Parameters:
          data: The data to use for tracing the module, either a dataloader (in which case
            the first batch is used) or a single batch of inputs.
          data_with_labels: Specifies whether the dataloader elements
            are (input, labels) pairs or just inputs (the default).

        Returns:
          self
        """
        if self.logdir is not None:
            batch = next(iter(data)) if isinstance(data, torch.utils.data.DataLoader) else data
            xs = validate_batch_input(batch, with_labels=data_with_labels)
            xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
            writer = self.get_tb_writer("train")
            writer.add_graph(self, xs)
            writer.flush()
        return self

    def log_console(self, message: str, end: str = "\n", file=None) -> Self:
        """Log the given message to the console, correctly even if the progress bar is being used."""
        tqdm.tqdm.write(message, file=file, end=end)

    def get_log_file(self) -> TextIO:
        """Possibly create and return a text-based log file for the current log.

        To use this method, nonempty `logdir` must have been set in `configure`.

        Returns:
          file: The opened log file.
        """
        assert self.logdir is not None, "Cannot use get_log_file when logdir is not set."
        if self._log_file is None:
            self._log_file = open(os.path.join(self.logdir, "logs.txt"), "a", encoding="utf-8")
        return self._log_file

    def get_tb_writer(self, name: str) -> torch.utils.tensorboard.SummaryWriter:
        """Possibly create and return a TensorBoard writer for the given name.

        To use this method, nonempty `logdir` must have been set in `configure`.

        Returns:
          writer: The opened TensorBoard writer.
        """
        assert self.logdir is not None, "Cannot use get_tb_writer when logdir is not set."
        if name not in self._tb_writers:
            self._tb_writers[name] = torch.utils.tensorboard.SummaryWriter(os.path.join(self.logdir, name))
        return self._tb_writers[name]
