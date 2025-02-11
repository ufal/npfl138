# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import os
import sys
from typing import Protocol, TextIO, TypeAlias

import numpy as np
import torch

Tensor: TypeAlias = torch.Tensor | torch.nn.utils.rnn.PackedSequence
TensorOrTensors: TypeAlias = Tensor | tuple[Tensor, ...] | list[Tensor]

Logs: TypeAlias = dict[str, float]


class LossProtocol(Protocol):
    def __call__(self, y_pred: TensorOrTensors, y: TensorOrTensors) -> torch.Tensor:
        ...


class MetricProtocol(Protocol):
    def reset(self) -> None:
        ...
    def update(self, y_pred: TensorOrTensors, y: TensorOrTensors) -> None:  # noqa: E301
        ...
    def compute(self) -> torch.Tensor:  # noqa: E301
        ...


class CallbackProtocol(Protocol):
    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> None:
        ...


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
    return x.numpy() if as_numpy else x


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
    batch: TensorOrTensors | tuple[TensorOrTensors, TensorOrTensors] | list[TensorOrTensors], with_output: bool,
) -> TensorOrTensors:
    if with_output:
        assert isinstance(batch, (tuple, list)) and len(batch) == 2, "The batch must be an (input, _) pair."
        batch = batch[0]
    assert check_tensors(batch), "The batch input must be a tensor or a tuple of tensors."
    return batch


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
        self.register_buffer("total", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

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
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    from time import time as _time
    from tqdm import tqdm as _tqdm

    def __init__(self, module: torch.nn.Module | None = None):
        """Initialize the module with the given PyTorch model.

        This constructor is useful when you want to wrap an existing module
        (e.g., a torch.nn.Sequential or a pretrained Transformer).
        """
        super().__init__()
        if module is not None:
            self._module = module
            self.forward = lambda *args, **kwargs: self._module(*args, **kwargs)

    def configure(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        loss: LossProtocol | None = None,
        metrics: dict[str, MetricProtocol] | None = None,
        initial_epoch: int | None = None,
        logdir: str | None = None,
        device: torch.device | str = "auto",
    ) -> None:
        """Configure the module fitting, evaluation, and placement.

        The method can be called multiple times, preserving previously set values for Nones.
        - `optimizer` is the optimizer to use for training;
        - `scheduler` is an optional learning rate scheduler used after every batch;
        - `loss` is the loss function to minimize;
        - `metrics` is a dictionary of additional metrics to compute, each being an object
          implementing the MetricProtocol (reset/update/compute), e.g., a torchmetrics.Metric;
        - `initial_epoch` is the initial epoch of the model used during training and evaluation;
        - `logdir` is an optional directory where TensorBoard logs should be written;
        - `device` is the device to move the module to; when "auto", the previously set
          device is kept, otherwise the first of cuda/mps/xpu is used if available.
        """
        self.optimizer = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        self.scheduler = scheduler if scheduler is not None else getattr(self, "scheduler", None)
        self.loss = loss if loss is not None else getattr(self, "loss", None)
        self.loss_tracker = getattr(self, "loss_tracker", LossTracker())
        self.metrics = torch.nn.ModuleDict(metrics or {}) \
            if metrics is not None or not hasattr(self, "metrics") else self.metrics
        self.epoch = initial_epoch if initial_epoch is not None else getattr(self, "epoch", 0)
        self._log_file, self._tb_writers = getattr(self, "_log_file", None), getattr(self, "_tb_writers", {})
        if logdir is not None and logdir != getattr(self, "logdir", None):  # reset loggers on a new logdir
            self._log_file, self._tb_writers = None, {}
        self.logdir = logdir if logdir is not None else getattr(self, "logdir", None)
        self.device = getattr(self, "device", get_auto_device()) if device == "auto" else torch.device(device)
        self.to(self.device)

    def save_weights(self, path: str, optimizer_path: str | None = None) -> None:
        """Save the model weights to the given path.

        If optimizer_path is given, the optimizer state is also saved,
        to a separate checkpoint, relative to the model weights path.
        """
        state_dict = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        if optimizer_path is not None:
            optimizer_state = {"epoch": self.epoch}
            self.optimizer is not None and optimizer_state.update(optimizer=self.optimizer.state_dict())
            self.scheduler is not None and optimizer_state.update(scheduler=self.scheduler.state_dict())
            optimizer_path = os.path.join(os.path.dirname(path), optimizer_path)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            torch.save(optimizer_state, optimizer_path)

    def load_weights(self, path: str, optimizer_path: str | None = None, device: torch.device | str = "auto") -> None:
        """Load the model weights from the given path.

        If the optimizer_path is given, the optimizer state is also loaded,
        with the optimizer_path resolved relative to the model weights path.
        The device specifies where to load the model to; when "auto", the previously set
        device is kept, otherwise the first of cuda/mps/xpu is used if available.
        """
        self.device = getattr(self, "device", get_auto_device()) if device == "auto" else torch.device(device)
        self.load_state_dict(torch.load(path, map_location=self.device))
        if optimizer_path is not None:
            optimizer_path = os.path.join(os.path.dirname(path), optimizer_path)
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.epoch = optimizer_state["epoch"]
            "optimizer" in optimizer_state and self.optimizer.load_state_dict(optimizer_state["optimizer"])
            "scheduler" in optimizer_state and self.scheduler.load_state_dict(optimizer_state["scheduler"])
        self.to(self.device)

    @staticmethod
    def save_config(config: dict, path: str) -> None:
        """Save a JSON-serializable configuration to the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_config(path: str) -> dict:
        """Load a JSON-serializable configuration from the given path."""
        with open(path, "r", encoding="utf-8-sig") as config_file:
            return json.load(config_file)

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        dev: torch.utils.data.DataLoader | None = None,
        callbacks: list[CallbackProtocol] = [],
        console: int = 3,
    ) -> Logs:
        """Train the model on the given dataset.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `console` controls the console verbosity: 0 for silent, 1 for epoch logs, 2 for
          additional only-when-writing-to-console progress bar, 3 for persistent progress bar.
        """
        logs, epochs = {}, self.epoch + epochs
        while self.epoch < epochs:
            self.epoch += 1
            self.train()
            self.loss_tracker.reset()
            for metric in self.metrics.values():
                metric.reset()
            start = self._time()
            epoch_message = f"Epoch {self.epoch}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if console == 2 else console < 2)
            for batch in data_and_progress:
                xs, y = validate_batch_input_output(batch)
                xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
                y = tuple(y_.to(self.device) for y_ in y) if is_sequence(y) else y.to(self.device)
                logs = self.train_step(xs, y)
                if not data_and_progress.disable:
                    logs_message = [f"{k}={v:#.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                    data_and_progress.set_description(f"{epoch_message} {logs_message}", refresh=False)
            logs = {f"train_{k}": v for k, v in logs.items()}
            if dev is not None:
                logs |= {f"dev_{k}": v for k, v in self.evaluate(dev, log_as=None).items()}
            for callback in callbacks:
                callback(self, self.epoch, logs)
            self.log_metrics(logs, epochs, self._time() - start, console)
        return logs

    def train_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single training step, returning the logs."""
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

    def compute_loss(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors) -> torch.Tensor:
        """Compute the loss of the model given the inputs, predictions, and target outputs."""
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: TensorOrTensors) -> Logs:
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        for metric in self.metrics.values():
            metric.update(y_pred, y)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        log_as: str | None = "test",
        callbacks: list[CallbackProtocol] = [],
        console: int = 1,
    ) -> Logs:
        """An evaluation of the model on the given dataset.

        - `dataloader` is the dataset to evaluate on, each element a pair of inputs
          and an output, the inputs either a single tensor or a tuple of tensors;
        - `log_as` is the name of the dataset used in the logs; when None, no logs are written;
        - `callbacks` is a list of callbacks to call after the evaluation with
          arguments `self`, `epoch`, and `logs`;
        - `console` controls the console verbosity: 0 for silent, 1 for a single message.
        """
        self.eval()
        self.loss_tracker.reset()
        for metric in self.metrics.values():
            metric.reset()
        start = self._time()
        for batch in dataloader:
            xs, y = validate_batch_input_output(batch)
            xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
            y = tuple(y_.to(self.device) for y_ in y) if is_sequence(y) else y.to(self.device)
            logs = self.test_step(xs, y)
        if log_as is not None:
            logs = {f"{log_as}_{k}": v for k, v in logs.items()}
            self.log_metrics(logs, elapsed=self._time() - start, console=console)
        return logs

    def test_step(self, xs: TensorOrTensors, y: TensorOrTensors) -> Logs:
        """An overridable method performing a single evaluation step, returning the logs."""
        with torch.no_grad():
            y_pred = self(*xs)
            loss = self.compute_loss(y_pred, y, *xs)
            return {"loss": self.loss_tracker(loss)} | self.compute_metrics(y_pred, y, *xs)

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataloader_with_outputs: bool = False,
        as_numpy: bool = True,
    ) -> list[torch.Tensor | tuple[torch.Tensor, ...] | np.ndarray | tuple[np.ndarray, ...]]:
        """Compute predictions for the given dataset.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `dataloader_with_outputs` specifies whether the dataloader elements
          are (input, output) pairs or just inputs (default);
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed.
        """
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = validate_batch_input(batch, with_output=dataloader_with_outputs)
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            y = self.predict_step(xs, as_numpy=as_numpy)
            predictions.extend(y if not isinstance(y, tuple) else zip(*y))
        return predictions

    def predict_step(
        self, xs: TensorOrTensors, as_numpy: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | np.ndarray | tuple[np.ndarray, ...]:
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            y = self(*xs)
            return maybe_unpack(y, as_numpy) if not is_sequence(y) else tuple(maybe_unpack(y_, as_numpy) for y_ in y)

    def log_metrics(
        self, logs: Logs, epochs: int | None = None, elapsed: float | None = None, console: int = 1,
    ) -> None:
        """Log the given dictionary to file logs, TensorBoard logs, and optionally the console."""
        if self.logdir is not None:
            for key, value in logs.items():
                writer, metric = key.split("_", maxsplit=1)
                self.get_tb_writer(writer).add_scalar(metric, value, self.epoch)
            for writer in dict.fromkeys(key.split("_", maxsplit=1)[0] for key in logs):
                self.get_tb_writer(writer).flush()
        for file in ([self.get_log_file()] if self.logdir is not None else []) + [sys.stdout] * bool(console):
            print(f"Epoch {self.epoch}" + (f"/{epochs}" if epochs is not None else ""),
                  *[f"{elapsed:.1f}s"] if elapsed is not None else [],
                  *[f"{k}={v:#.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()], file=file, flush=True)

    def log_config(self, config: dict, sort_keys: bool = True, console: int = 1) -> None:
        """Log the given dictionary to the file logs, TensorBoard logs, and optionally the console."""
        if self.logdir is not None:
            config = dict(sorted(config.items())) if sort_keys else config
            writer = self.get_tb_writer("train")
            writer.add_text("config", json.dumps(config, ensure_ascii=False, indent=2), self.epoch)
            writer.flush()
        for file in ([self.get_log_file()] if self.logdir is not None else []) + [sys.stdout] * bool(console):
            print("Config", f"epoch={self.epoch}", *[f"{k}={v}" for k, v in config.items()], file=file, flush=True)

    def get_log_file(self) -> TextIO:
        if self._log_file is None:
            self._log_file = open(os.path.join(self.logdir, "logs.txt"), "a", encoding="utf-8")
        return self._log_file

    def get_tb_writer(self, name: str) -> torch.utils.tensorboard.SummaryWriter:
        """Possibly create and return a TensorBoard writer for the given name."""
        if name not in self._tb_writers:
            self._tb_writers[name] = self._SummaryWriter(os.path.join(self.logdir, name))
        return self._tb_writers[name]
