#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO: Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = ...

        # TODO: Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`. During the model computation, the layer will
        # process the word embeddings generated by the `self._word_embedding` layer,
        # and we will sum the outputs of forward and backward directions.
        self._word_rnn = ...

        # TODO: Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = ...

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO: Start by embedding the `word_ids` using the word embedding layer.
        hidden = ...

        # TODO: Process the embedded words through the RNN layer. Because the sentences
        # have different length, you have to use `torch.nn.utils.rnn.pack_padded_sequence`
        # to construct a variable-length `PackedSequence` from the input. You need to compute
        # the length of each sentence in the batch (by counting non-`MorphoDataset.PAD` tokens);
        # note that these lengths must be on CPU, so you might need to use the `.cpu()` method.
        # Finally, also pass `batch_first=True` and `enforce_sorted=False` to the call.
        packed = ...

        # TODO: Pass the `PackedSequence` through the RNN, choosing the appropriate output.
        packed = ...

        # TODO: Unpack the RNN output using the `torch.nn.utils.rnn.pad_packed_sequence` with
        # `batch_first=True` argument. Then sum the outputs of forward and backward directions.
        hidden = ...

        # TODO: Pass the RNN output through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = ...

        return hidden


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = ...
        tag_ids = ...
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO: Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = ...
        # TODO: Process `tag_ids` analogously to `word_ids`.
        tag_ids = ...
        return word_ids, tag_ids


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

    # Load the data.
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO: Create the Adam optimizer.
        optimizer=...,
        # TODO: Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation. Note that the loss
        # expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=...,
        # TODO: Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
        # the first argument, `num_classes` set to the number of unique tags, and
        # again `ignore_index=morpho.PAD` to ignore the padded tags.
        metrics={"accuracy": torchmetrics.Accuracy(...)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
