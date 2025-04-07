#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--clip_gradient", default=None, type=float, help="Norm for gradient clipping.")
parser.add_argument("--dev_sequences", default=1_000, type=int, help="Number of development sequences.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=0, type=int, help="Additional hidden layer after RNN.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU", "RNN"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=10, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_sequences", default=10_000, type=int, help="Number of training sequences.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Dataset for generating sequences of random integers in the range [0, sequence_dim - 1],
# with labels indicating the parity of the sum of the integers in the sequence.
class ParitySequences:
    def __init__(self, sequences_num: int, sequence_length: int, sequence_dim: int, seed: int) -> None:
        sequences = torch.zeros([sequences_num, sequence_length, sequence_dim], dtype=torch.int64)
        labels = torch.zeros([sequences_num, sequence_length, 1], dtype=torch.int64)
        generator = torch.Generator().manual_seed(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = torch.randint(0, max(2, sequence_dim), size=[sequence_length], generator=generator)
            labels[i, :, 0] = torch.bitwise_and(sequences[i, :, 0].cumsum(0), 1)
            if sequence_dim > 1:
                sequences[i] = torch.nn.functional.one_hot(sequences[i, :, 0], sequence_dim)
        self._sequences = sequences.to(torch.float32)
        self._labels = labels.to(torch.float32)

    @property
    def sequences(self) -> torch.Tensor:
        return self._sequences

    @property
    def labels(self) -> torch.Tensor:
        return self._labels


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # Construct the required layers.

        # TODO: The sequence will be processed using an RNN with type `args.rnn` (LSTM/GRU/RNN)
        # and with dimensionality `args.rnn_dim`.
        ...

        # TODO: If `args.hidden_layer` is nonzero, the result of the RNN should be processed
        # by a fully connected layer with `args.hidden_layer` units and ReLU activation.
        ...

        # TODO: The predictions are generated using a fully connected output layer
        # with one output and sigmoid activation.
        ...

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Process the input sequence through the RNN and the other layers.
        ...


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

    # Create the data.
    train = ParitySequences(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42)
    dev = ParitySequences(args.dev_sequences, args.sequence_length, args.sequence_dim, seed=43)

    train = torch.utils.data.TensorDataset(train.sequences, train.labels)
    dev = torch.utils.data.TensorDataset(dev.sequences, dev.labels)

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size)

    # Create the model and train
    model = Model(args)

    optimizer = torch.optim.Adam(model.parameters())

    if args.clip_gradient is not None:
        def gradient_clipping(optimizer, _args, _kwargs):
            # TODO: Implement gradient clipping using `torch.nn.utils.clip_grad_norm_`,
            # clipping the gradient if its L2 norm is larger than `args.clip_gradient`.
            ...
        optimizer.register_step_pre_hook(gradient_clipping)

    model.configure(
        optimizer=optimizer,
        loss=torch.nn.BCELoss(),
        metrics={"accuracy": torchmetrics.Accuracy("binary")},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
