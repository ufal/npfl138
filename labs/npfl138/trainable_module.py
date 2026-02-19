# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The Keras-inspired high-level API for training PyTorch models.

The [TrainableModule][npfl138.TrainableModule] class is
a high-level API for training PyTorch models. It is a subclass of
[torch.nn.Module][] and:

- It provides a high-level API for training, evaluation, and prediction
  via [fit][npfl138.TrainableModule.fit], [evaluate][npfl138.TrainableModule.evaluate],
  and [predict][npfl138.TrainableModule.predict] methods. Each can be customized
  by overriding the corresponding [train_step][npfl138.TrainableModule.train_step],
  [test_step][npfl138.TrainableModule.test_step], or [predict_step][npfl138.TrainableModule.predict_step]
  methods.

- The module automatically handles moving the module to a specified device,
  using the first available accelerator by default. To this end,
  [configure][npfl138.TrainableModule.configure] or
  [load_weights][npfl138.TrainableModule.load_weights] must always be called
  first before using the high-level API.

- The module inputs and outputs can be either single tensors or _tensor structures_
  (the [TensorOrTensors][npfl138.type_aliases.TensorOrTensors] type), where the latter
  can be tuples, lists, or dictionaries containing other tensor structures and non-tensor
  values, or completely custom data structures. Tensors in tuples, lists, and dictionary
  values are automatically moved to the correct device during [fit][npfl138.TrainableModule.fit],
  [evaluate][npfl138.TrainableModule.evaluate], and [predict][npfl138.TrainableModule.predict].

- The module provides API for serialization and deserialization of the model,
  both the weights ([save_weights][npfl138.TrainableModule.save_weights],
  [load_weights][npfl138.TrainableModule.load_weights]) and the model options
  ([save_options][npfl138.TrainableModule.save_options],
  [load_options][npfl138.TrainableModule.load_options]).

- The module keeps a collection of tracked [losses][npfl138.TrainableModule.losses], and
  a collection of [metrics][npfl138.TrainableModule.metrics] implementing the
  [Metric][npfl138.Metric] interface (e.g., any our metric or any metric from `torchmetric` package),
  and is capable of storing the computed logs in text files, TensorBoard logs, WandB, or any
  implementation of the [Logger][npfl138.Logger] interface.
"""
import argparse
from collections.abc import Iterable
import functools
import json
import os
import sys
import time
from typing import Any, Literal, Self

import torch
import torchmetrics

from .callback import Callback, STOP_TRAINING
from .logger import Logger
from .loggers import BaseLogger, FileSystemLogger, MultiLogger, TensorBoardLogger
from .loss import Loss
from .metric import Metric
from .metrics import Mean
from .progress_logger import ProgressLogger as BaseProgressLogger
from .type_aliases import Logs, TensorOrTensors
from .utils import compute_logs, fill_and_standardize_path, tuple_list


class KeepPrevious:
    pass
keep_previous = KeepPrevious()  # noqa: E305


class ProgressLogger(BaseProgressLogger):
    @staticmethod
    def log_config(config: dict[str, Any], epoch: int, console: int | None, logger: Logger | None) -> None:
        if ProgressLogger.get_console_verbosity(console) >= 1:
            print(BaseLogger.format_config_as_text(config, epoch), flush=True)
        logger and logger.log_config(config, epoch)

    def log_epoch(self, logs: dict[str, float], epoch: int, elapsed: float, logger: Logger | None = None) -> None:
        description = self._description + f" {elapsed:.1f}s"
        self._console and print(description, BaseLogger.format_metrics(logs), flush=True)
        logger and logger.log_metrics(logs, epoch, description)


def get_auto_device() -> torch.device:
    """Return an available accelerator or CPU if none is available, unless overridden by `NPFL_DEVICE`."""
    if "NPFL_DEVICE" in os.environ:
        return torch.device(os.environ["NPFL_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class static_or_instance_method:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner=None):
        return functools.partial(self.func, instance)


def tensors_concatenate(x: list[TensorOrTensors] | tuple[TensorOrTensors, ...]) -> TensorOrTensors:
    """Concatenate a list of tensors or tensor structures along the first dimension."""
    if not x:
        raise RuntimeError("Cannot concatenate an empty list of tensors.")
    first = x[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(x)
    elif isinstance(first, torch.nn.utils.rnn.PackedSequence):
        raise RuntimeError("Concatenation of torch.nn.utils.rnn.PackedSequence is not supported.")
    elif isinstance(first, tuple):
        return tuple(map(tensors_concatenate, zip(*x)))
    elif isinstance(first, list):
        return [tensors_concatenate(b) for b in zip(*x)]
    elif isinstance(first, dict):
        return {k: tensors_concatenate([b[k] for b in x]) for k in first.keys()}
    raise RuntimeError(f"Cannot concatenate tensors of type {type(first)}.")


def tensors_to_device(x: TensorOrTensors, device: torch.device) -> TensorOrTensors:
    """Asynchronously move the input tensor or the input tensor structure to the given device.

    Parameters:
      x: The input tensor or tensor structure to move to the device, where tensor structures
        can be tuples, lists, or dictionaries containing other tensor structures and non-tensor
        values, or completely custom data structures. All tensors in tuples, lists, and dictionary
        values are moved.
      device: The device to move the tensors to.

    Returns:
      The input tensor or tensor structure with all tensors moved to the given device.
    """
    if isinstance(x, (torch.Tensor, torch.nn.utils.rnn.PackedSequence)):
        return x.to(device, non_blocking=True)
    elif isinstance(x, tuple):
        return tuple(tensors_to_device(a, device) for a in x)
    elif isinstance(x, list):
        return [tensors_to_device(a, device) for a in x]
    elif isinstance(x, dict):
        return {k: tensors_to_device(v, device) for k, v in x.items()}
    return x


def tensors_to_device_as_tuple(x: TensorOrTensors, device: torch.device) -> tuple[TensorOrTensors, ...]:
    """Asynchronously move input tensor data structure to a given device, returning a (possibly one-element) tuple."""
    if isinstance(x, (torch.Tensor, torch.nn.utils.rnn.PackedSequence)):
        return (x.to(device),)
    elif isinstance(x, tuple_list):
        return tuple(tensors_to_device(a, device) for a in x)
    elif isinstance(x, dict):
        return ({k: tensors_to_device(v, device) for k, v in x.items()},)
    else:
        return (x,)


def tensors_to_numpy(x: TensorOrTensors) -> TensorOrTensors:
    """Convert tensors in a given input tensor data structure to NumPy arrays."""
    if isinstance(x, torch.Tensor):
        return x.numpy(force=True)
    elif isinstance(x, torch.nn.utils.rnn.PackedSequence):
        raise RuntimeError("Conversion of torch.nn.utils.rnn.PackedSequence to Numpy is not supported.")
    elif isinstance(x, tuple):
        return tuple(tensors_to_numpy(b) for b in x)
    elif isinstance(x, list):
        return [tensors_to_numpy(b) for b in x]
    elif isinstance(x, dict):
        return {k: tensors_to_numpy(v) for k, v in x.items()}
    return x


def validate_batch_input_output(
    batch: tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors],
) -> tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors]:
    """Validate that the given batch is an (input, output) pair and return it."""
    assert isinstance(batch, tuple_list) and len(batch) == 2, "The batch must be an (input, output) pair."
    return batch


def validate_batch_input(
    batch: TensorOrTensors | tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors], with_labels: bool,
) -> TensorOrTensors:
    """If with_labels is True, validate that the given batch is an (input, _) pair and return the input."""
    if with_labels:
        assert isinstance(batch, tuple_list) and len(batch) == 2, "The batch must be an (input, _) pair."
        batch = batch[0]
    return batch


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

    def __init__(self, module: torch.nn.Module | None = None):
        """Initialize the module, optionally with an existing PyTorch module.

        Parameters:
          module: An optional existing PyTorch module to wrap, e.g., a [torch.nn.Sequential][]
            or a pretrained Transformer. If given, the module still must be configured.
        """
        super().__init__()
        setattr(self, "device", None)  # Avoid mkdocs documenting the attribute before its definition.
        setattr(self, "module", module)  # Avoid mkdocs documenting the attribute before its definition.
        self.unconfigure()
        if module is not None:
            self.forward = self._wrapped_module_forward

    def _wrapped_module_forward(self, inputs):
        return self.module(inputs)

    def configure(
        self,
        *,
        optimizer: torch.optim.Optimizer | None | KeepPrevious = keep_previous,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None | KeepPrevious = keep_previous,
        loss: Loss | None | KeepPrevious = keep_previous,
        metrics: dict[str, Metric | torchmetrics.Metric] | KeepPrevious = keep_previous,
        initial_epoch: int | KeepPrevious = keep_previous,
        logdir: str | None | KeepPrevious = keep_previous,
        loggers: list[Logger] | KeepPrevious = keep_previous,
        device: torch.device | str | Literal["auto"] | KeepPrevious = keep_previous,
    ) -> Self:
        """Configure the module fitting, evaluation, and placement.

        The method can be called multiple times, preserving previously set values by default.

        Note:
          The [self.device][npfl138.TrainableModule.device], [self.epoch][npfl138.TrainableModule.epoch],
          [self.logger][npfl138.TrainableModule.logger] and [self.metrics][npfl138.TrainableModule.metrics]
          fields are never `None` after this call (they are the only options for which `None` is not allowed).

        Parameters:
          optimizer: The optimizer to use for training.
          scheduler: An optional learning rate scheduler used after every batch.
          loss: The loss function to minimize, implementing the
            [Loss][npfl138.Loss] protocol.
          metrics: A dictionary of additional metrics to compute, each being an
            object implementing the metric (reset/update/compute) interface, e.g.,
            any [Metric][npfl138.Metric] or any `torchmetrics.Metric`.
          initial_epoch: The initial epoch of the model used during training and evaluation.
          logdir: An optional directory where textual logs should be stored; when specified,
            [npfl138.loggers.FileSystemLogger][] is added to the loggers.
          loggers: An optional list of loggers to use for logging; when not specified and
            `logdir` is given, [npfl138.loggers.TensorBoardLogger][] is added to the loggers
            (specifying a list of loggers including `[]` prevents that).
          device: The device to move the module to. When `"auto"`, or `keep_previous`
            with no previously set device, an accelerator [torch.accelerator.current_accelerator][]
            is used if available, otherwise a CPU; furthermore, environment variable `NPFL_DEVICE`
            can be used to override the "auto" device selection.

        Returns:
          self
        """
        self.optimizer = optimizer if optimizer is not keep_previous else self.optimizer
        self.scheduler = scheduler if scheduler is not keep_previous else self.scheduler
        self.loss = loss if loss is not keep_previous else self.loss
        if metrics is not keep_previous or self.metrics is None:
            self.metrics = torch.nn.ModuleDict({} if metrics is keep_previous else metrics)
        self.epoch = initial_epoch if initial_epoch is not keep_previous else self.epoch or 0
        if (logdir is not keep_previous and logdir != self.logdir) or loggers is not keep_previous:
            self.logger = self.logger and self.logger.close()
        if (logdir is not keep_previous and logdir != self.logdir) and logdir:
            loggers = [TensorBoardLogger(logdir)] if loggers is keep_previous else loggers
            loggers = [FileSystemLogger(logdir)] + loggers
        loggers = [] if self.logger is None and loggers is keep_previous else loggers
        self.logger = MultiLogger(loggers) if loggers is not keep_previous else self.logger
        self.logdir = logdir if logdir is not keep_previous else self.logdir
        if device is not keep_previous or self.device is None:
            self.device = get_auto_device() if device == "auto" or device is keep_previous else torch.device(device)
        return self.to(self.device)

    def unconfigure(self) -> Self:
        """Remove all training configuration of the TrainableModule.

        Only the module device is kept.

        Returns:
          self
        """
        getattr(self, "logger", None) is not None and self.logger.close()
        self.loss, self.optimizer, self.scheduler, self.epoch = None, None, None, None
        self.logdir, self.logger, self.losses, self.metrics = None, None, None, None
        return self

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        *,
        dev: torch.utils.data.DataLoader | None = None,
        callbacks: list[Callback] = [],
        log_config: dict[str, Any] = {},
        log_graph: bool = False,
        console: int | None = None,
    ) -> Logs:
        """Train the model on the given dataset.

        Note:
          To customize the training, you can override the [train_step][npfl138.TrainableModule.train_step],
          [compute_loss][npfl138.TrainableModule.compute_loss], and/or
          [compute_metrics][npfl138.TrainableModule.compute_metrics] method.

        Parameters:
          dataloader: The training dataset, each element a pair of inputs and outputs;
            the inputs and outputs can be either single tensors or tensor structures.
          epochs: The number of epochs to train.
          dev: An optional development dataset to evaluate after every epoch, with the
            same format as the training dataset.
          callbacks: A list of callbacks to call after every epoch, each implementing
            the [npfl138.Callback][] protocol with arguments `self`, `epoch`, and `logs`,
            possibly returning [npfl138.STOP_TRAINING][] to stop the training (note that
            the module is set to evaluation mode before calling each callback).
          log_config: Optionally log the given configuration dictionary at the beginning of training.
          log_graph: Optionally log the model graph at the beginning of training.
          console: Controls the console verbosity: 0 for silent, 1 for epoch logs, 2 for
            additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `NPFL_PROGRESS` environment variable.

        Returns:
          logs: A dictionary of logs from the training and optionally dev evaluation; the logs are
            fully evaluated to just float values.

        Note:
          The module is set to evaluation mode when returning from this method.
        """
        assert self.metrics is not None, "The TrainableModule has not been configured, run configure first."
        log_config and ProgressLogger.log_config(log_config, self.epoch, console, self.logger)
        logs, epochs, stop_training = {}, self.epoch + epochs, False
        while self.epoch < epochs and not stop_training:
            self.epoch += 1
            self.train()
            self.losses = torch.nn.ModuleDict()
            for metric in self.metrics.values():
                metric.reset()
            start, logs = time.time(), {}
            data_with_progress = ProgressLogger(dataloader, f"Epoch {self.epoch}/{epochs}", console, lambda: logs)
            for batch in data_with_progress:
                xs, y = validate_batch_input_output(batch)
                xs = tensors_to_device_as_tuple(xs, self.device)
                y = tensors_to_device(y, self.device)
                logs = self.train_step(xs, y)
                log_graph = log_graph and self.logger.log_graph(self.module or self, xs, self.epoch - 1) and False
            if dev is not None:
                compute_logs(logs)
                logs |= self.eval().evaluate(dev, "dev", log_results=False, console=console)
            for callback in callbacks:
                stop_training = callback(self.eval(), self.epoch, compute_logs(logs)) is STOP_TRAINING or stop_training
            data_with_progress.log_epoch(compute_logs(logs), self.epoch, time.time() - start, self.logger)
        self.eval()
        return logs

    def train_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single training step, returning the logs.

        Parameters:
          xs: The input batch to the model, either a single tensor or a tensor structure.
          y: The target output batch of the model, either a single tensor or a tensor structure.

        Returns:
          logs: A dictionary of logs from the training step.
        """
        y_pred = self(*xs)
        loss = self.track_loss(self.compute_loss(y_pred, y, *xs))
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler is not None and self.scheduler.step()
            metrics = self.compute_metrics(y_pred, y, *xs)
            return {**({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {}), **self.losses, **metrics}

    def compute_loss(
        self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the loss of the model given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a tensor structure.
          y: The target output of the model, either a single tensor or a tensor structure.
          *xs: The inputs to the model; unpacked, if the input is a list or a tuple.

        Returns:
          loss: The computed loss, either a single tensor or a dictionary of named loss components.
            In case of a dictionary, the total loss is either the item called `"loss"` if present,
            or the sum of all items otherwise.
        """
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors) -> Logs:
        """Compute and return metrics given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a tensor structure.
          y: The target output of the model, either a single tensor or a tensor structure.
          *xs: The inputs to the model; unpacked, if the input is a list or a tuple.

        Returns:
          logs: A dictionary of computed metrics.
        """
        for metric in self.metrics.values():
            metric.update(y_pred, y)
        return self.metrics

    def track_loss(self, loss: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """A method for manually tracking a loss value during training or evaluation.

        Parameters:
          loss: The loss value to track, either a single tensor or a dictionary of named loss components.
            In case of a dictionary, the total loss is either the item called `"loss"` if present,
            or the sum of all items otherwise.

        Returns:
          total_loss: The total loss tensor.
        """
        if isinstance(loss, dict):
            if "loss" not in loss:
                loss = {"loss": sum(loss.values())} | loss
        else:
            loss = {"loss": loss}

        for name, value in loss.items():
            if name not in self.losses:
                self.losses[name] = Mean(device=self.device)
            self.losses[name].update(value)

        return loss["loss"]

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str | None = "test",
        *,
        log_results: bool = True,
        callbacks: list[Callback] = [],
        console: int | None = None,
    ) -> Logs:
        """An evaluation of the model on the given dataset.

        Note:
          To customize the evaluation, you can override the [test_step][npfl138.TrainableModule.test_step],
          [compute_loss][npfl138.TrainableModule.compute_loss], and/or
          [compute_metrics][npfl138.TrainableModule.compute_metrics] method.

        Parameters:
          dataloader: The dataset to evaluate on, each element a pair of inputs and outputs;
            the inputs and outputs can be either single tensors or tensor structures.
          dataset_name: An optional name of the dataset used as a prefix of the metric names in the logs.
          log_results: If `True` (the default), the evaluation results are logged using the module logger,
            and they are also printed (if `console` is not 0); if `False`, they are just returned.
          callbacks: A list of callbacks to call after the evaluation, each implementing
            the [npfl138.Callback][] protocol with arguments `self`, `epoch`, and `logs`.
          console: Controls the console verbosity: 0 for silent, 1 for a single message,
            2 for additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `NPFL_PROGRESS` environment variable.

        Returns:
          logs: A dictionary of logs from the evaluation, each name prefixed with `f"{dataset_name}:"`
            if `dataset_name` is given; the logs are fully evaluated to just float values.

        Note:
          The module is set to evaluation mode when returning from this method.
        """
        assert self.metrics is not None, "The TrainableModule has not been configured, run configure first."
        self.eval()
        self.losses = torch.nn.ModuleDict()
        for metric in self.metrics.values():
            metric.reset()
        start, logs = time.time(), {}
        data_with_progress = ProgressLogger(dataloader, "Evaluation", console, lambda: logs)
        for batch in data_with_progress:
            xs, y = validate_batch_input_output(batch)
            xs = tensors_to_device_as_tuple(xs, self.device)
            y = tensors_to_device(y, self.device)
            logs = self.test_step(xs, y)
            logs = {f"{dataset_name}:{k}": v for k, v in logs.items()} if dataset_name else logs
        for callback in callbacks:
            callback(self.eval(), self.epoch, compute_logs(logs))
        compute_logs(logs)
        log_results and data_with_progress.log_epoch(logs, self.epoch, time.time() - start, self.logger)
        self.eval()
        return logs

    def test_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single evaluation step, returning the logs.

        Parameters:
          xs: The input batch to the model, either a single tensor or a tensor structure.
          y: The target output batch of the model, either a single tensor or a tensor structure.

        Returns:
          logs: A dictionary of logs from the evaluation step.
        """
        with torch.no_grad():
            y_pred = self(*xs)
            self.track_loss(self.compute_loss(y_pred, y, *xs))
            metrics = self.compute_metrics(y_pred, y, *xs)
            return {**self.losses, **metrics}

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        data_with_labels: bool = False,
        whole_batches: bool = False,
        as_numpy: bool = False,
        console: int | None = None,
    ) -> Iterable[TensorOrTensors]:
        """Compute predictions for the given dataset as a generator of individual items or whole batches.

        None:
          To customize the prediction, you can override the [predict_step][npfl138.TrainableModule.predict_step]
          and/or [unpack_batch][npfl138.TrainableModule.unpack_batch] methods.

        Parameters:
          dataloader: The dataset to predict on, each element either directly the module
            input or a pair whose first element is the input; the input can be either
            a single tensor or a tensor structure.
          data_with_labels: Specifies whether the dataloader elements are _(input, labels)_ pairs
            or just _inputs_ (the default).
          whole_batches: If `False` (the default), the predicted batches are unpacked into individual items
            using the [unpack_batch][npfl138.TrainableModule.unpack_batch] method; if `True`, whole predicted
            batches are yielded.
          as_numpy: If `False` (the default), the predicted tensors are kept as PyTorch tensors on the module device;
            if `True`, they are converted to Numpy arrays.
          console: Controls the console verbosity: 0 and 1 for silent, 2 for
            additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `NPFL_PROGRESS` environment variable.

        Returns:
          predictions: An iterable whose elements are the individual predicted items.
        """
        assert self.device is not None, "No device has been set for the TrainableModule, run configure first."
        self.eval()
        for batch in ProgressLogger(dataloader, "Prediction", console):
            xs = validate_batch_input(batch, with_labels=data_with_labels)
            xs = tensors_to_device_as_tuple(xs, self.device)
            y = self.predict_step(xs)
            y = self.unpack_batch(y, *xs) if not whole_batches else [y]
            yield from map(tensors_to_numpy, y) if as_numpy else y

    def predict_step(self, xs: TensorOrTensors) -> TensorOrTensors:
        """An overridable method performing a single prediction step.

        Parameters:
          xs: The input batch to the model, either a single tensor or a tensor structure.

        Returns:
          predictions: The predicted batch.
        """
        with torch.no_grad():
            return self(*xs)

    def unpack_batch(self, y: TensorOrTensors, *xs: TensorOrTensors) -> Iterable[TensorOrTensors]:
        """An overridable method unpacking a batch of predictions into individual items.

        The default implementation handles batches of single [torch.Tensor][]s,
        [torch.nn.utils.rnn.PackedSequence][]s, or tuples, lists, and dictionaries
        containing elements (values in the case of dictionaries) that are themselves
        unpackable by this method.

        Parameters:
          y: The batch predictions, either a single tensor or a tensor structure.
          *xs: The inputs to the model; unpacked, if the input is a list or a tuple.

        Returns:
          items: An iterable over the individual predicted items.
        """
        if isinstance(y, torch.Tensor):
            yield from y
        elif isinstance(y, torch.nn.utils.rnn.PackedSequence):
            yield from torch.nn.utils.rnn.unpack_sequence(y)
        elif isinstance(y, tuple):
            yield from zip(*(self.predicted_batch_as_items(b) for b in y))
        elif isinstance(y, list):
            yield from map(list, zip(*(self.predicted_batch_as_items(b) for b in y)))
        elif isinstance(y, dict):
            for items in zip(*(self.predicted_batch_as_items(v) for v in y.values())):
                yield dict(zip(y.keys(), items))
        else:
            raise RuntimeError(f"Cannot unpack batch of type {type(y)} into individual items.")

    def predict_batch(self, xs: TensorOrTensors, *, as_numpy: bool = False) -> TensorOrTensors:
        """Run prediction on a single batch, returning the predicted batch.

        This method is a convenience wrapper around [predict_step][npfl138.TrainableModule.predict_step].
        It sets the module to evaluation mode, move the input to the module device, calls
        [predict_step][npfl138.TrainableModule.predict_step], and optionally converts the output to Numpy arrays.

        None:
          To customize prediction, you can override the [predict_step][npfl138.TrainableModule.predict_step] method.

        Warning:
          To avoid calling [torch.nn.Module.eval][] too often, `predict_batch` calls [torch.nn.Module.eval][]
          only if `self.training` is `True`.

        Parameters:
          xs: The input batch to the model, either a single tensor or a tensor structure. Note that it
            cannot be a pair of inputs and outputs.
          as_numpy: If `False` (the default), the predicted tensors are kept as PyTorch tensors on the module device;
            if `True`, they are converted to Numpy arrays.

        Returns:
          predictions: The predicted batch.
        """
        assert self.device is not None, "No device has been set for the TrainableModule, run configure first."
        self.training and self.eval()
        xs = tensors_to_device_as_tuple(xs, self.device)
        y = self.predict_step(xs)
        y = tensors_to_numpy(y) if as_numpy else y
        return y

    def predict_tensor(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        data_with_labels: bool = False,
        as_numpy: bool = False,
        console: int | None = None,
    ) -> TensorOrTensors:
        """Compute predictions for the given dataset, returning them concatenated as a single tensor/tensor structure.

        This method is a convenience wrapper around [predict][npfl138.TrainableModule.predict].

        None:
          To customize prediction, you can override the [predict_step][npfl138.TrainableModule.predict_step] method.

        Parameters:
          dataloader: The dataset to predict on, each element either directly the module
            input or a pair whose first element is the input; the input can be either
            a single tensor or a tensor structure.
          data_with_labels: Specifies whether the dataloader elements are _(input, labels)_ pairs
            or just _inputs_ (the default).
          as_numpy: If `False` (the default), the predicted tensors are kept as PyTorch tensors on the module device;
            if `True`, they are converted to Numpy arrays.
          console: Controls the console verbosity: 0 and 1 for silent, 2 for
            additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `NPFL_PROGRESS` environment variable.

        Returns:
          predictions: The predicted dataset concatenated to a single tensor or a tensor structure.
        """
        y = list(self.predict(dataloader, data_with_labels=data_with_labels, whole_batches=True, console=console))
        y = tensors_concatenate(y)
        return tensors_to_numpy(y) if as_numpy else y

    def save_weights(self, path: str, optimizer_path: str | None = None) -> Self:
        """Save the model weights to the given path.

        Both the `path` and `optimizer_path` can contain `{logdir}` and `{epoch}` placeholders.
        They are always processed with `str.format` before use, and both slashes and backslashes
        are replaced with the current OS path separator.

        Parameters:
          path: The path to save the model weights to; a `.pt` extension is recommended.
          optimizer_path: An optional path to save the optimizer state to, relative to the
            model weights path.

        Returns:
          self
        """
        path = fill_and_standardize_path(path, logdir=self.logdir, epoch=self.epoch)
        if optimizer_path is not None:
            optimizer_path = fill_and_standardize_path(optimizer_path, logdir=self.logdir, epoch=self.epoch)
        os.path.dirname(path) and os.makedirs(os.path.dirname(path), exist_ok=True)

        state_dict = self.state_dict()
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

        Both the `path` and `optimizer_path` can contain `{logdir}` and `{epoch}` placeholders.
        They are always processed with `str.format` before use, and both slashes and backslashes
        are replaced with the current OS path separator.

        Parameters:
          path: The path to load the model weights from.
          optimizer_path: An optional path to load the optimizer state from, relative to the
            model weights path.
          device: The device to load the module to. When `"auto"`, or `keep_previous`
            with no previously set device, an accelerator [torch.accelerator.current_accelerator][]
            is used if available, otherwise a CPU; furthermore, environment variable `NPFL_DEVICE`
            can be used to override the "auto" device selection.

        Returns:
          self
        """
        path = fill_and_standardize_path(path, logdir=self.logdir, epoch=self.epoch)
        if optimizer_path is not None:
            optimizer_path = fill_and_standardize_path(optimizer_path, logdir=self.logdir, epoch=self.epoch)

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
        return self.to(self.device)

    @static_or_instance_method
    def save_options(self, _path_: str, /, **kwargs: Any) -> None:
        """Save a JSON-serializable options or [argparse.Namespace][]s to the given path.

        The method can be called both as a static method and as an instance method.

        When called as an instance method, the path can contain `{logdir}` and `{epoch}` placeholders.
        The path is always processed with `str.format` before use, and both slashes
        and backslashes are replaced with the current OS path separator.

        Parameters:
          _path_: The path to save the options to; a `.json` extension is recommended. The parameter
            name uses the underscores to avoid conflict with possible option `path`.
          **kwargs: The option values to save.
        """
        path = fill_and_standardize_path(_path_, **{"logdir": self.logdir, "epoch": self.epoch} if self else {})
        options = dict((k + ": argparse.Namespace", vars(v)) if isinstance(v, argparse.Namespace) else (k, v)
                       for k, v in kwargs.items())
        os.path.dirname(path) and os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as options_file:
            json.dump(options, options_file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_options(path: str) -> dict:
        """Load a JSON-serializable options or [argparse.Namespace][]s from the given path.

        Parameters:
          path: The path to load the options from.

        Returns:
          options: The loaded options dictionary.
        """
        with open(path, "r", encoding="utf-8-sig") as options_file:
            options = json.load(options_file)
        return dict((k.removesuffix(": argparse.Namespace"), argparse.Namespace(**v))
                    if k.endswith(": argparse.Namespace") else (k, v) for k, v in options.items())

    def profile(
        self,
        steps: int,
        export_trace: str | None = None,
        *,
        warmup: int | None = 0,
        lightweight: bool = False,
        export_memory_timeline: str | bool = False,
        export_cuda_allocations: str | bool = False,
        quit_when_done: bool = False,
    ) -> None:
        """Profile the module execution for a number of steps.

        Run the PyTorch profiler on a CPU and an accelerator if available (and optionally track CUDA allocations),
        for the given number of steps (forward passes) after an optional number of warmup steps.

        By default, the profiler records detailed information including shapes, stack traces, and memory utilization,
        which has additional overhead. When `lightweight=True`, only basic information is recorded.

        Info:
          The exported profile trace can be inspected in TensorBoard using the `torch-tb-profiler` plugin that
          can be installed using `pip install torch-tb-profiler`.

        Info:
          The CUDA memory allocations snapshot can be inspected using the <https://docs.pytorch.org/memory_viz> page.

        Parameters:
          steps: The number of steps (forward calls) to profile. For example, when `steps=2`, the profiler starts
            at the beginning of the first step (forward) and stops at the beginning of the third step (forward).
          export_trace: An optional path to export the main profile to (as a Chrome trace JSON file). The file must
            have an extension of either `.pt.trace.json` or `.pt.trace.json.gz` to be recognized by the
            `torch-tb-profiler` plugin; if the path does not end with one of those extensions, `.pt.trace.json.gz`
            is appended.
          warmup: An optional number of warmup steps to skip before starting the profiling.

            - When 0 (the default), profiling starts at the beginning of the first step (forward call).
            - When 1, profiling starts at the beginning of the second step (forward call).
            - When `None`, the profiling starts immediately (which can be useful to track CUDA allocations
              during module initialization).
          lightweight: If `True`, use a lightweight profiling mode that records only basic information, excluding
            tensor shapes, stack traces, and memory utilization. This reduces the profiling overhead.
          export_memory_timeline: An optional path to export the memory timeline HTML report to. If a string is
            passed, it is used as the path (appending `.html` if needed); if `True` is passed, the path is derived
            from `export_trace` by replacing the extension with `.html`.

            **Note**: Requires the `matplotlib` package for generating the graph.
          export_cuda_allocations: An optional path to export the CUDA memory allocations snapshot to (when CUDA
            is available). If a string is passed, it is used as the path (appending `.pickle` if needed);
            if `True` is passed, the path is derived from `export_trace` by replacing the extension with `.pickle`.
          quit_when_done: If `True`, the program exits when profiling is done.
        """
        if lightweight and export_memory_timeline is not False:
            raise ValueError("The export_memory_timeline argument cannot be used with lightweight=True.")

        # Standardize all export paths.
        if export_trace is not None:
            if not export_trace.endswith(".pt.trace.json") and not export_trace.endswith(".pt.trace.json.gz"):
                export_trace += ".pt.trace.json.gz"
            export_trace = fill_and_standardize_path(export_trace, logdir=self.logdir)

        if isinstance(export_memory_timeline, str):
            if not export_memory_timeline.endswith(".html"):
                export_memory_timeline += ".html"
            export_memory_timeline = fill_and_standardize_path(export_memory_timeline, logdir=self.logdir)
        elif export_memory_timeline is True:
            assert export_trace is not None, "export_trace must be specified when export_memory_timeline is True."
            export_memory_timeline = export_trace.rsplit(".pt.trace.json", maxsplit=1)[0] + ".html"

        if isinstance(export_cuda_allocations, str):
            if not export_cuda_allocations.endswith(".pickle"):
                export_cuda_allocations += ".pickle"
            export_cuda_allocations = fill_and_standardize_path(export_cuda_allocations, logdir=self.logdir)
        elif export_cuda_allocations is True:
            assert export_trace is not None, "export_trace must be specified when export_cuda_allocations is True."
            export_cuda_allocations = export_trace.rsplit(".pt.trace.json", maxsplit=1)[0] + ".pickle"

        # Set up the profiler hook.
        profiler, hook = None, None

        def profile_step(_module, _inputs) -> None:
            nonlocal steps, warmup, profiler
            if warmup > 0:
                warmup -= 1
            elif steps > 0:
                if profiler is None:
                    if export_cuda_allocations and torch.cuda.is_available():
                        torch.cuda.memory._record_memory_history()
                    # We use `acc_events=True` to avoid false-positive warning in PyTorch 2.10 about acc_events.
                    profiler = torch.profiler.profile(
                        profile_memory=not lightweight, record_shapes=not lightweight, with_stack=not lightweight,
                        acc_events=True)
                    profiler.__enter__()
                steps -= 1
            elif profiler is not None:
                profiler.__exit__(None, None, None)
                if export_cuda_allocations and torch.cuda.is_available():
                    torch.cuda.memory._dump_snapshot(export_cuda_allocations)
                    torch.cuda.memory._record_memory_history(enabled=None)
                if export_trace:
                    profiler.export_chrome_trace(export_trace)
                if export_memory_timeline:
                    profiler.export_memory_timeline(export_memory_timeline)
                hook.remove()
                profiler = None
                quit_when_done and sys.exit(0)

        # Register the profiler hook.
        hook = self.register_forward_pre_hook(profile_step)

        # When warmup is `None`, start profiling immediately.
        if warmup is None:
            warmup = 0
            steps += 1
            profile_step(None, None)

    def log_console(
        self, message: str, end: str = "\n", progress_only: bool = False, console: int | None = None,
    ) -> Self:
        """Write the given message to the console, correctly even if a progress bar is being used.

        Parameters:
          message: The message to write.
          end: The string appended after the message.
          progress_only: If `False` (the default), the message is written to standard output when current console
            verbosity is at least 1; if `True`, the message is written to standard error only when the progress bar
            is being shown (console verbosity 2 and writing to the console, or console verbosity 3).
          console: Controls the current console verbosity. The default is 2, but can be overridden by the
            `NPFL_PROGRESS` environment variable.
        """
        ProgressLogger.log_console(message, end, progress_only, console)
        return self

    def get_tb_writer(self, name: str) -> torch.utils.tensorboard.SummaryWriter:
        """Possibly create and return a TensorBoard writer for the given name.

        To use this method, a [npfl138.loggers.TensorBoardLogger][] must have been
        created in the [configure][npfl138.TrainableModule.configure] method, either by
        specifying a `logdir` and no `loggers` or by explicitly passing the TensorBoard logger.

        Returns:
          writer: The opened TensorBoard writer.
        """
        for logger in self.logger.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.get_writter(name)
        raise RuntimeError("No TensorBoardLogger found in the configured loggers.")

    device: torch.device | None
    """The device where the module is located, if already set."""

    epoch: int | None
    """The current one-based epoch of the module, if already set."""

    logdir: str | None
    """The directory where the logs are stored, if set."""

    loss: Loss | None
    """The loss function used for training, if set."""

    logger: MultiLogger | None
    """The logger used for logging the training and evaluation, if set."""

    losses: torch.nn.ModuleDict | None
    """The dictionary of tracked losses used in training and evaluation, if set."""

    metrics: torch.nn.ModuleDict | None
    """The dictionary of metrics used for training and evaluation, if set."""

    module: torch.nn.Module | None
    """The wrapped PyTorch module, if specified during construction."""

    optimizer: torch.optim.Optimizer | None
    """The optimizer used for training, if set."""

    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    """The learning rate scheduler used for training, if set."""
