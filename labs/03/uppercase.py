#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.3")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=..., type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=..., type=int, help="Window size to use.")


class Dataset(torch.utils.data.Dataset):
    # A dataset must always implement at least `__len__` and `__getitem__`.
    def __init__(self, uppercase_dataset: UppercaseData.Dataset):
        self.windows = uppercase_dataset.windows
        self.labels = uppercase_dataset.labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index], self.labels[index]

    # When a dataset implements `__getitems__`, this method is used to generate whole batches in a single call.
    # However, `__getitems__` is expected to return a list of items that are later collated together.
    # For maximum speedup, we already return a whole batch from `__getitems__` and implement a trivial `collate`.
    def __getitems__(self, indices):
        indices = torch.as_tensor(indices)
        return self.windows[indices], self.labels[indices]

    @staticmethod
    def collate(batch):
        return batch


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.
        ...

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        ...


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    args.logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data and create windows of integral character indices and integral labels.
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    train = torch.utils.data.DataLoader(
        Dataset(uppercase_data.train), args.batch_size, collate_fn=Dataset.collate, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(uppercase_data.dev), args.batch_size, collate_fn=Dataset.collate)
    test = torch.utils.data.DataLoader(Dataset(uppercase_data.test), args.batch_size, collate_fn=Dataset.collate)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    model = ...

    # TODO: Generate correctly capitalized test set and write the result to `predictions_file`,
    # which is by default `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        # We start by generating the network test set predictions; if you modified the `test` dataloader
        # or your model does not process the dataset windows, you might need to adjust the following line.
        predictions = model.predict(test, data_with_labels=True)

        # Now you need to utilize the network predictions and the unannotated test data (lowercased text)
        # available in `uppercase_data.test.text` to produce capitalized text and print it to the `predictions_file`.
        ...


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
