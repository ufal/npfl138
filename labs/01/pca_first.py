#!/usr/bin/env python3
import argparse

import numpy as np
import torch

import npfl138
from npfl138.datasets.mnist import MNIST
npfl138.require_version("2526.1")

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
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    # Prepare the data.
    mnist = MNIST()

    data_indices = np.random.choice(len(mnist.train), size=args.examples, replace=False)
    data = mnist.train.data["images"][data_indices].to(torch.float32) / 255

    # TODO: Data has shape [args.examples, MNIST.C, MNIST.H, MNIST.W].
    # We want to reshape it to [args.examples, MNIST.C * MNIST.H * MNIST.W].
    # We can do so using `torch.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = ...

    # TODO: Now compute mean of every feature. Use `torch.mean`, and set
    # `dim` (or `axis`) argument to zero -- therefore, the mean will be
    # computed across the first dimension, so across examples.
    #
    # Note that for compatibility with Numpy/TF/Keras, all `dim` arguments
    # in PyTorch can be also called `axis`.
    mean = ...

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T @ (data - mean) / data.shape[0]
    # where transpose can be computed using `torch.transpose` or `torch.t` and
    # matrix multiplication using either Python operator @ or `torch.matmul`.
    cov = ...

    # TODO: Compute the total variance, which is the sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `torch.diagonal`,
    # and to sum a tensor use `torch.sum`.
    total_variance = ...

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type `torch.float32` using `torch.ones`.
    v = ...
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov v
        #    The matrix-vector multiplication can be computed as regular matrix multiplication
        #    or using `torch.mv`.
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using for example `torch.linalg.vector_norm`.
        # 3. v = v / s
        ...

    # The `v` is now approximately the eigenvector of the largest eigenvalue, `s`.
    # We now compute the explained variance, which is the ratio of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance, 100 * explained_variance


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(main_args)
    print(f"Total variance: {total_variance:.2f}")
    print(f"Explained variance: {explained_variance:.2f}%")
