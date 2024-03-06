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

    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    accuracy = keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(args.epochs):
        accuracy.reset_state()
        for batch in mnist.train.batches(args.batch_size):
            probabilities = model(batch["images"], training=True)
            loss = loss_fn(batch["labels"], probabilities)
            accuracy(batch["labels"], probabilities)

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                optimizer.apply([v.value.grad for v in model.trainable_variables], model.trainable_variables)
        train = accuracy.result()

        accuracy.reset_state()
        for batch in mnist.dev.batches(args.batch_size):
            probabilities = model(batch["images"], training=False)
            accuracy(batch["labels"], probabilities)
        dev = accuracy.result()
        print("Epoch {} finished, train: {}, dev: {}".format(epoch + 1, train, dev))

    accuracy.reset_state()
    for batch in mnist.test.batches(args.batch_size):
        probabilities = model(batch["images"], training=False)
        accuracy(batch["labels"], probabilities)
    test = accuracy.result()
    print("Test: {}".format(test))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
