#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.2")
from npfl138 import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_layer_size, MNIST.LABELS),
    )

    # Wrap the model in the TrainableModule.
    model = npfl138.TrainableModule(model)

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`) with
    # the given `args.learning_rate`.
    # - For `SGD`, if `args.momentum` is specified, use Nesterov momentum.
    # - If `args.decay` is set, then also create a LR scheduler (otherwise, pass `None`).
    #   The scheduler should decay the learning rate from the initial `args.learning_rate`
    #   to the final `args.learning_rate_final`. The `scheduler.step()` is called after
    #   each batch, so the number of scheduler iterations is the number of batches in all
    #   training epochs (note that `len(train)` is the number of batches in one epoch).
    #   - for `linear`, use `torch.optim.lr_scheduler.LinearLR` and set `start_factor`,
    #     `end_factor`, and `total_iters` appropriately;
    #   - for `exponential`, use `torch.optim.lr_scheduler.ExponentialLR` and set `gamma`
    #     appropriately (be careful to compute it using float64 to avoid precision issues);
    #   - for `cosine`, use `torch.optim.lr_scheduler.CosineAnnealingLR` and set `T_max`
    #     and `eta_min` appropriately.
    #   In all cases, you should reach `args.learning_rate_final` just after the training.
    #
    #   If a learning rate schedule is used, the `TrainableModule` automatically logs the
    #   learning rate to the console and to TensorBoard. Additionally, you can find out
    #   the next learning rate to be used by printing `model.scheduler.get_last_lr()[0]`.
    #   Therefore, after the training, this value should be `args.learning_rate_final`.
    ...

    model.configure(
        optimizer=...,
        scheduler=...,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    # Train the model.
    logs = model.fit(train, dev=dev, epochs=args.epochs)

    if args.decay:
        print("Next learning rate to be used: {:g}".format(model.scheduler.get_last_lr()[0]))

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
