#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.utils.tensorboard

import npfl138
npfl138.require_version("2425.2")
from npfl138 import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = torch.nn.Parameter(
            torch.randn(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer) * 0.1,
            requires_grad=True,  # This is the default.
        )
        self._b1 = torch.nn.Parameter(torch.zeros(args.hidden_layer))

        # TODO: Create the rest of the parameters:
        # - _W2, which is a parameter of size `[args.hidden_layer, MNIST.LABELS]`,
        #   initialized to `torch.randn` value with standard deviation 0.1,
        # - _b2, which is a parameter of size `[MNIST.LABELS]` initialized to zeros.
        self._W2 = ...
        self._b2 = ...

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Define the computation of the network. Notably:
        # - start by casting the input uint8 images to float32 using `.to(torch.float32)`,
        # - then divide the tensor by 255 to normalize it to the `[0, 1]` range,
        # - then reshape it to the shape `[inputs.shape[0], -1]`;
        #   the -1 is a wildcard which is computed so that the number
        #   of elements before and after the reshape is preserved,
        # - then multiply it by `self._W1` and then add `self._b1`,
        # - apply `torch.tanh`,
        # - finally, multiply the result by `self._W2` and then add `self._b2`.
        return ...

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        self.train()
        for batch in dataset.batches(self._args.batch_size, shuffle=True):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.C, MNIST.H, MNIST.W]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # TODO: Start by moving the batch data to the device where the model is.
            # This is needed, because the data is currently on CPU, but the model might
            # be on a GPU. You can move the data using the `.to(device)` method, and you
            # can obtain the device of the model using for example `self._W1.device`.
            images = batch["images"].to(...)
            labels = batch["labels"].to(...)

            # TODO: Compute the predicted logits of the batch images by calling `self(...)`.
            logits = ...

            # TODO: Compute the probabilities of the batch images using `torch.softmax`.
            probabilities = ...

            # TODO: Manually compute the loss:
            # - For every batch example, the loss is the categorical crossentropy of the
            #   predicted probabilities and the gold label. To compute the crossentropy, you can
            #   - either use `torch.nn.functional.one_hot` to obtain one-hot encoded gold labels
            #     and then compute the crossentropy from the definition, or
            #   - you can start by obtaining directly the model probabilities of the gold labels
            #     by suitably indexing the predicted probabilities by the gold labels.
            #   Note that it might be necessary to convert the labels from bytes to `torch.int64`.
            # - Finally, compute the average across the batch examples.
            loss = ...

            # We create a list of all parameters. Note that a `torch.nn.Module` automatically
            # tracks owned parameters, so we could also use `list(self.parameters())`.
            parameters = [self._W1, self._b1, self._W2, self._b2]

            # TODO: Compute the gradient of the loss with respect to parameters using
            # the backpropagation algorithm, by
            # - first resetting the gradients of all parameters to zero with `self.zero_grad()`,
            # - then calling `loss.backward()`.
            ...

            gradients = [parameter.grad for parameter in parameters]
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients):
                    # TODO: Perform the SGD update with learning rate `self._args.learning_rate`
                    # for the parameter and computed gradient. You can modify the
                    # parameter value directly with `-=`, or you can use `parameter.sub_`.
                    ...

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self.eval()
        with torch.no_grad():
            # Compute the accuracy of the model prediction
            correct = 0
            for batch in dataset.batches(self._args.batch_size):
                # TODO: Compute the logits of the batch images as in the training,
                # and then convert them to Numpy with `.numpy(force=True)`.
                logits = ...

                # TODO: Evaluate how many batch examples were predicted
                # correctly and increase `correct` variable accordingly, assuming
                # the model predicts the class with the highest logit/probability.
                correct += ...

        return correct / len(dataset)


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load raw data.
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    # Try using an accelerator if available.
    if torch.cuda.is_available():
        model = model.to(device="cuda")
    elif torch.mps.is_available():
        model = model.to(device="mps")
    elif torch.xpu.is_available():
        model = model.to(device="xpu")

    for epoch in range(args.epochs):
        # TODO: Run the `train_epoch` with `mnist.train` dataset
        ...

        # TODO: Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        dev_accuracy = ...
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy), flush=True)
        writer.add_scalar("dev/accuracy", 100 * dev_accuracy, epoch + 1)

    # TODO: Evaluate the test data using `evaluate` on `mnist.test` dataset
    test_accuracy = ...
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
