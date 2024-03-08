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
parser.add_argument("--model_type", default="sequential", choices=["sequential", "functional", "subclassing"])
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model
    if args.model_type == "sequential":
        model = keras.Sequential()
        model.add(keras.Input([MNIST.H, MNIST.W, MNIST.C]))
        model.add(keras.layers.Rescaling(1 / 255))
        model.add(keras.layers.Flatten())
        for hidden_layer in args.hidden_layers:
            model.add(keras.layers.Dense(hidden_layer, activation="relu"))
        model.add(keras.layers.Dense(MNIST.LABELS, activation="softmax"))
        model.summary()

    elif args.model_type == "functional":
        inputs = keras.Input([MNIST.H, MNIST.W, MNIST.C])
        hidden = keras.layers.Rescaling(1 / 255)(inputs)
        hidden = keras.layers.Flatten()(hidden)
        for hidden_layer in args.hidden_layers:
            hidden = keras.layers.Dense(hidden_layer, activation="relu")(hidden)
        outputs = keras.layers.Dense(MNIST.LABELS, activation="softmax")(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

    elif args.model_type == "subclassing":
        class Model(keras.Model):
            def __init__(self, hidden_layers):
                super().__init__()

                self.rescaling_layer = keras.layers.Rescaling(1 / 255)
                self.flatten_layer = keras.layers.Flatten()
                self.hidden_layers = [keras.layers.Dense(hidden_layer, activation="relu")
                                      for hidden_layer in hidden_layers]
                self.output_layer = keras.layers.Dense(MNIST.LABELS, activation="softmax")

            def call(self, inputs):
                hidden = self.rescaling_layer(inputs)
                hidden = self.flatten_layer(hidden)
                for hidden_layer in self.hidden_layers:
                    hidden = hidden_layer(hidden)
                return self.output_layer(hidden)

        model = Model(args.hidden_layers)

    else:
        raise ValueError("Unknown model type '{}'".format(args.model_type))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    )

    model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
