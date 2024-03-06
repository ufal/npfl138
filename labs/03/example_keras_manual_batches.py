#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model
    inputs = keras.Input([MNIST.H, MNIST.W, MNIST.C])
    hidden = keras.layers.Rescaling(1 / 255)(inputs)
    hidden = keras.layers.Flatten()(hidden)
    for hidden_layer in args.hidden_layers:
        hidden = keras.layers.Dense(hidden_layer, activation="relu")(hidden)
    outputs = keras.layers.Dense(MNIST.LABELS, activation="softmax")(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    for epoch in range(args.epochs):
        model.reset_metrics()
        for batch in mnist.train.batches(args.batch_size):
            train = model.train_on_batch(batch["images"], batch["labels"], return_dict=True)

        model.reset_metrics()
        for batch in mnist.dev.batches(args.batch_size):
            dev = model.test_on_batch(batch["images"], batch["labels"], return_dict=True)
        print("Epoch {} finished.\n  Train: {}\n  Dev: {}".format(epoch + 1, train, dev))

    model.reset_metrics()
    for batch in mnist.test.batches(args.batch_size):
        test = model.test_on_batch(batch["images"], batch["labels"], return_dict=True)
    print("Test: {}".format(test))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
