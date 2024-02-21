#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(mnist.train.size, size=args.examples, replace=False)
    data = keras.ops.convert_to_tensor(mnist.train.data["images"][data_indices] / 255, dtype="float32")

    # TODO: Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    # We can do so using `keras.ops.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = ...

    # TODO: Now compute mean of every feature. Use `keras.ops.mean`, and set
    # `axis` to zero -- therefore, the mean will be computed across the first
    # dimension, so across examples.
    mean = ...

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    # where transpose can be computed using `keras.ops.transpose` and
    # matrix multiplication using either Python operator @ or `keras.ops.matmul`.
    cov = ...

    # TODO: Compute the total variance, which is the sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `keras.ops.diagonal`,
    # and to sum a tensor use `keras.ops.sum`.
    total_variance = ...

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type `"float32"` using `keras.ops.ones`.
    v = ...
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov v
        #    The matrix-vector multiplication can be computed as regular matrix multiplication.
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using for example `keras.ops.norm`.
        # 3. v = v / s
        pass

    # The `v` is now approximately the eigenvector of the largest eigenvalue, `s`.
    # We now compute the explained variance, which is the ratio of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance, 100 * explained_variance


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(explained_variance))
