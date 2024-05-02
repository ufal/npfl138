#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics

from morpho_dataset import MorphoDataset
from trainable_module import TrainableModule

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(TrainableModule):
    class FFN(torch.nn.Module):
        def __init__(self, dim, expansion):
            super().__init__()
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            raise NotImplementedError()

        def forward(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            raise NotImplementedError()

    class SelfAttention(torch.nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O; each a module parameter
            # `torch.nn.Parameter` of shape `[dim, dim]`. The weights should be initialized
            # using the `torch.nn.init.xavier_uniform_`.
            raise NotImplementedError()

        def forward(self, inputs, mask):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `torch.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - permute dimensions via `torch.permute` to `[batch_size, heads, max_sentence_len, dim // heads]`.

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (nonzero value) or padding (zero value).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `torch.exp(-1e9)` is also zero because of limited precision).

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - permute the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            raise NotImplementedError()

    class PositionalEmbedding(torch.nn.Module):
        def forward(self, inputs):
            # TODO: Compute the sinusoidal positional embeddings. Assuming the embeddings have
            # a shape `[max_sentence_len, dim]` with `dim` even, and for `0 <= i < dim/2`:
            # - the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, dim/2 + i]` should be
            #     `cos(pos / 10_000 ** (2 * i / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            raise NotImplementedError()

    class Transformer(torch.nn.Module):
        def __init__(self, layers, dim, expansion, heads, dropout):
            super().__init__()
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.

        def forward(self, inputs, mask):
            # TODO: First compute the positional embeddings.

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            raise NotImplementedError()

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the form ids
        # from `train.forms.word_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = ...

        # TODO: Create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        self._transformer = ...

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.word_vocab` is the tag vocabulary.
        self._output_layer = ...

        # Initialize the layers using the Keras-inspired initialization. You can try
        # removing this line to see how much worse the default PyTorch initialization is.
        self.apply(self.keras_init)

    def forward(self, form_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `form_ids` using the word embedding layer.
        hidden = ...

        # TODO: Process the embedded forms through the transformer. As the second argument,
        # pass the attention mask `form_ids != MorphoDataset.PAD`.
        hidden = ...

        # TODO(tagger_we): Pass `hidden` through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimension.
        hidden = ...

        return hidden


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    def prepare_tagging_data(example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input forms as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `word_vocab` of `morpho.train.forms` and `morpho.train.tags`.
        form_ids = ...
        tag_ids = ...
        return form_ids, tag_ids
    train = morpho.train.transform(prepare_tagging_data)
    dev = morpho.dev.transform(prepare_tagging_data)

    def prepare_batch(data):
        # Construct a single batch, where `data` is a list of examples
        # generated by `prepare_tagging_data`.
        form_ids, tag_ids = zip(*data)
        # TODO(tagger_we): Combine `form_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        form_ids = ...
        # TODO(tagger_we): Process `tag_ids` analogously to `form_ids`.
        tag_ids = ...
        return form_ids, tag_ids
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, collate_fn=prepare_batch, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, batch_size=args.batch_size, collate_fn=prepare_batch)

    model.configure(
        # TODO(tagger_we): Create the optimizer by creating an instance of
        # `torch.optim.Adam` which will train the `model.parameters()`.
        optimizer=...,
        # TODO(tagger_we): Use `torch.nn.CrossEntropyLoss` to instantiate the loss function.
        # Pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation. Note that the loss
        # expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=...,
        # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
        # the first argument, `num_classes` set to the number of unique tags, and
        # again `ignore_index=morpho.PAD` to ignore the padded tags.
        metrics={"accuracy": torchmetrics.Accuracy(...)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development and training losses for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
