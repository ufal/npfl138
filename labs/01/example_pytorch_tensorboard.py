#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
from npfl138 import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


def main(args: argparse.Namespace) -> None:
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
    test = torch.utils.data.DataLoader(Dataset(mnist.test), batch_size=args.batch_size)

    # Create the model.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_layer_size, MNIST.LABELS),
    )
    print("The following model has been created:", model)

    # Create the TrainableModule and configure it for training.
    model = npfl138.TrainableModule(model)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )
    model.get_tb_writer("train").add_graph(model, torch.zeros(1, MNIST.C, MNIST.H, MNIST.W, device=model.device))

    # Train the model.
    model.fit(train, dev=dev, epochs=args.epochs)

    # Evaluate the model on the test data.
    model.evaluate(test)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
