#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torch.utils.tensorboard

from mnist import MNIST

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


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = keras.Variable(
            keras.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed),
            trainable=True,
        )
        self._b1 = keras.Variable(keras.ops.zeros([args.hidden_layer]), trainable=True)

        # TODO(sgd_backpropagation): Create variables:
        # - _W2, which is a trainable variable of size `[args.hidden_layer, MNIST.LABELS]`,
        #   initialized to `keras.random.normal` value `with stddev=0.1` and `seed=args.seed`,
        # - _b2, which is a trainable variable of size `[MNIST.LABELS]` initialized to zeros
        ...

    def predict(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by casting the input byte image to `float32` with `keras.ops.cast`
        # - then divide the tensor by 255 to normalize it to the `[0, 1]` range
        # - then reshape it to the shape `[inputs.shape[0], -1]`.
        #   The -1 is a wildcard which is computed so that the number
        #   of elements before and after the reshape is preserved.
        # - then multiply it by `self._W1` and then add `self._b1`
        # - apply `keras.ops.tanh`
        # - multiply the result by `self._W2` and then add `self._b2`
        # - finally apply `keras.ops.softmax` and return the result

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tanh, and the input layer after reshaping.
        return ..., ..., ...

    def train_epoch(self, dataset: MNIST.Datasplit) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # TODO: Contrary to `sgd_backpropagation`, the goal here is to compute
            # the gradient manually, without calling `.backward()`. ReCodEx disables
            # PyTorch automatic differentiation during evaluation.
            #
            # Compute the input layer, hidden layer and output layer
            # of the batch images using `self.predict`.

            # TODO: Compute the gradient of the loss with respect to all
            # variables. Note that the loss is computed as in `sgd_backpropagation`:
            # - For every batch example, the loss is the categorical crossentropy of the
            #   predicted probabilities and the gold label. To compute the crossentropy, you can
            #   - either use `keras.ops.one_hot` to obtain one-hot encoded gold labels,
            #   - or suitably use `keras.ops.take_along_axis` to "index" the predicted probabilities.
            # - Finally, compute the average across the batch examples.
            #
            # During the gradient computation, you will need to compute
            # a batched version of a so-called outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`,
            # which you can achieve by using for example
            #   `A[:, :, np.newaxis] * B[:, np.newaxis, :]`
            # or with
            #   `keras.ops.einsum("ai,aj->aij", A, B)`.

            # TODO(sgd_backpropagation): Perform the SGD update with learning rate `self._args.learning_rate`
            # for the variable and computed gradient. You can modify the
            # variable value with `variable.assign` or in this case the more
            # efficient `variable.assign_sub`.
            ...

    def evaluate(self, dataset: MNIST.Datasplit) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            # TODO: Compute the probabilities of the batch images using `self.predict`
            # and convert them to Numpy with `keras.ops.convert_to_numpy`.
            probabilities = ...

            # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            correct += ...

        return correct / dataset.size


def main(args: argparse.Namespace) -> tuple[float, float]:
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

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # TODO: Run the `train_epoch` with `mnist.train` dataset

        # TODO: Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = ...
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        writer.add_scalar("dev/accuracy", 100 * accuracy, epoch + 1)

    # TODO: Evaluate the test data using `evaluate` on `mnist.test` dataset
    test_accuracy = ...
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy), flush=True)
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
