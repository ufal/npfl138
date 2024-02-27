#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


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
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                self.add_logs("train", {"learning_rate": self.model.optimizer.learning_rate.numpy()}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


def main(args: argparse.Namespace) -> None:
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
        keras.layers.Input([MNIST.H, MNIST.W, MNIST.C]),
        keras.layers.Rescaling(1 / 255),
        keras.layers.Flatten(),
        keras.layers.Dense(args.hidden_layer, activation="relu"),
        keras.layers.Dense(MNIST.LABELS, activation="softmax"),
    ])
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = TorchTensorBoardCallback(args.logdir)
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    tb_callback.add_logs("test", test_logs, args.epochs)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
