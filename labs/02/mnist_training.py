#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    model = keras.Sequential([
        keras.layers.Rescaling(1 / 255),
        keras.layers.Flatten(),
        keras.layers.Dense(args.hidden_layer, activation="relu"),
        keras.layers.Dense(MNIST.LABELS, activation="softmax"),
    ])

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
    # - For `SGD`, if `args.momentum` is specified, use Nesterov momentum.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `learning_rate` argument.
    # - If `args.decay` is set, then
    #   - for `linear`, use `keras.optimizers.schedules.PolynomialDecay` with the
    #     default `power=1.0`, and set `end_learning_rate` appropriately;
    #     https://keras.io/api/optimizers/learning_rate_schedules/polynomial_decay/
    #   - for `exponential`, use `keras.optimizers.schedules.ExponentialDecay`,
    #     and set `decay_rate` appropriately (keep the default `staircase=False`);
    #     https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
    #   - for `cosine`, use `keras.optimizers.schedules.CosineDecay`,
    #     and set `alpha` appropriately;
    #     https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/
    #   - in all cases, you should reach the `args.learning_rate_final` just after the
    #     training, i.e., the first update after the training should use exactly the
    #     given `args.learning_rate_final`;
    #   - in all cases, `decay_steps` must be **the total number of optimizer updates**,
    #     i.e., the total number of training batches in all epochs. The size of
    #     the training MNIST dataset is `mnist.train.size`, and you can assume it
    #     is exactly divisible by `args.batch_size`.
    #   Pass the created `{Polynomial,Exponential,Cosine}Decay` to the optimizer
    #   using the `learning_rate` constructor argument.
    #
    #   If a learning rate schedule is used, TensorBoard automatically logs the last used learning
    #   rate value in every epoch. Additionally, you can find out the last used learning
    #   rate by printing `model.optimizer.learning_rate` (the original schedule is available
    #   in `model.optimizer._learning_rate` if needed), so after training, the learning rate
    #   should be `args.learning_rate_final`.

    model.compile(
        optimizer=...,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = TorchTensorBoardCallback(args.logdir)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
