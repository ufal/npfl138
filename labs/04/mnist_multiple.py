#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import numpy as np
import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - keras.layers.Rescaling(1 / 255) to convert images to floats in [0, 1] range,
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature vector FV of each image.

        # TODO: Using the computed representations, the model should produce four outputs:
        # - first, compute _direct comparison_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations FV,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with "sigmoid" activation
        # - then, classify the computed representation FV of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation FV of the second image using
        #   the same layer (identical, i.e., with shared weights) into 10 classes;
        # - finally, compute _indirect comparison_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs; convert the comparison to "float32" using `keras.ops.cast`.
        outputs = {
            "direct_comparison": ...,
            "digit_1": ...,
            "digit_2": ...,
            "indirect_comparison": ...,
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Define the appropriate losses for the model outputs
        # "direct_comparison", "digit_1", "digit_2". Regarding metrics,
        # the accuracy of both the direct and indirect comparisons should be
        # computed; name both metrics "accuracy" (i.e., pass "accuracy" as the
        # first argument of the metric object).
        self.compile(
            optimizer=keras.optimizers.Adam(),
            loss={
                "direct_comparison": ...,
                "digit_1": ...,
                "digit_2": ...,
            },
            metrics={
                "direct_comparison": [...],
                "indirect_comparison": [...],
            },
        )

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace,
    ) -> torch.utils.data.Dataset:
        # Original MNIST dataset.
        images, labels = mnist_dataset.data["images"], mnist_dataset.data["labels"]

        # The new dataset should be created from consecutive _pairs_ of examples.
        # You can assume that the size of the original dataset is even.
        class TorchDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                # TODO: The new dataset has half the size of the original one.
                return ...

            def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
                # TODO: Given an `index`, generate a dataset element suitable for our model.
                # Notably, the element should be a pair `(input, output)`, with
                # - `input` being a pair of images `(images[2 * index], images[2 * index + 1])`,
                # - `output` being a dictionary with keys "digit_1", "digit_2", "direct_comparison",
                #   and "indirect_comparison".
                return ...

        return TorchDataset()


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable dataloaders from the MNIST data.
    train = torch.utils.data.DataLoader(model.create_dataset(mnist.train, args), args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(model.create_dataset(mnist.dev, args), args.batch_size)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev)

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
