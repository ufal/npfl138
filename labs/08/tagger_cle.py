#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")
# If you add more arguments, ReCodEx will keep them with your default values.


class TrainableModule(torch.nn.Module):
    """A simple Keras-like module for training with raw PyTorch.

    The module provides fit/evaluate/predict methods, computes loss and metrics,
    and generates both TensorBoard and console logs. By default, it uses GPU
    if available, and CPU otherwise. Additionally, it offers a Keras-like
    initialization of the weights.

    The current implementation supports models with either single input or
    a tuple of inputs; however, only one output is currently supported.
    """
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    from time import time as _time
    from tqdm import tqdm as _tqdm

    def configure(self, *, optimizer=None, schedule=None, loss=None, metrics={}, logdir=None, device="auto"):
        """Configure the module process.

        - `optimizer` is the optimizer to use for training;
        - `schedule` is an optional learning rate scheduler used after every batch;
        - `loss` is the loss function to minimize;
        - `metrics` is a dictionary of additional metrics to compute;
        - `logdir` is an optional directory where TensorBoard logs should be written;
        - `device` is the device to use; when "auto", `cuda` is used when available, `cpu` otherwise.
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss, self.loss_metric = loss, torchmetrics.MeanMetric()
        self.metrics = torchmetrics.MetricCollection(metrics)
        self.logdir, self._writers = logdir, {}
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.to(self.device)

    def load_weights(self, path, device="auto"):
        """Load the model weights from the given path."""
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self, path):
        """Save the model weights to the given path."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        """Train the model on the given dataset.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `verbose` controls the verbosity: 0 for silent, 1 for persistent progress bar,
          2 for a progress bar only when writing to a console.
        """
        for epoch in range(epochs):
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch+1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)
            for xs, y in data_and_progress:
                xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
                logs = self.train_step(xs, y)
                message = [epoch_message] + [f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                data_and_progress.set_description(" ".join(message), refresh=False)
            if dev is not None:
                logs |= {"dev_" + k: v for k, v in self.evaluate(dev, verbose=0).items()}
            for callback in callbacks:
                callback(self, epoch, logs)
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("dev_")}, epoch + 1)
            self.add_logs("dev", {k[4:]: v for k, v in logs.items() if k.startswith("dev_")}, epoch + 1)
            verbose and print(epoch_message, "{:.1f}s".format(self._time() - start),
                              *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def train_step(self, xs, y):
        """An overridable method performing a single training step.

        A dictionary with the loss and metrics should be returned."""
        self.zero_grad()
        y_pred = self.forward(*xs)
        loss = self.compute_loss(y_pred, y, *xs)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {}) \
                | self.compute_metrics(y_pred, y, *xs, training=True)

    def compute_loss(self, y_pred, y, *xs):
        """Compute the loss of the model given the inputs, predictions, and target outputs."""
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred, y, *xs, training):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        self.metrics.update(y_pred, y)
        return self.metrics.compute()

    def evaluate(self, dataloader, verbose=1):
        """An evaluation of the model on the given dataset.

        - `dataloader` is the dataset to evaluate on, each element a pair of inputs
          and an output, the inputs either a single tensor or a tuple of tensors;
        - `verbose` controls the verbosity: 0 for silent, 1 for a single message."""
        self.eval()
        self.loss_metric.reset()
        self.metrics.reset()
        for xs, y in dataloader:
            xs, y = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(self.device)
            logs = self.test_step(xs, y)
        verbose and print("Evaluation", *[f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def test_step(self, xs, y):
        """An overridable method performing a single evaluation step.

        A dictionary with the loss and metrics should be returned."""
        with torch.no_grad():
            y_pred = self.forward(*xs)
            self.loss_metric.update(self.compute_loss(y_pred, y, *xs))
            return {"loss": self.loss_metric.compute()} | self.compute_metrics(y_pred, y, *xs, training=False)

    def predict(self, dataloader, as_numpy=True):
        """Compute predictions for the given dataset.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed."""
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = batch[0] if isinstance(batch, tuple) else batch
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            predictions.extend(self.predict_step(xs, as_numpy=as_numpy))
        return predictions

    def predict_step(self, xs, as_numpy=True):
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            batch = self.forward(*xs)
            return batch.numpy(force=True) if as_numpy else batch

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = self._SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    @staticmethod
    def keras_init(module):
        """Initialize weights using the Keras defaults."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.uniform_(module.weight, -0.05, 0.05)
        if isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase)):
            for name, parameter in module.named_parameters():
                "weight_ih" in name and torch.nn.init.xavier_uniform_(parameter)
                "weight_hh" in name and torch.nn.init.orthogonal_(parameter)
                "bias" in name and torch.nn.init.zeros_(parameter)
                if "bias" in name and isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
                    parameter.data[module.hidden_size:module.hidden_size * 2] = 1


class Model(TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs):
            # Only mask during training and when the mask probability is non-zero.
            if self.training and self._mask_probability:
                # TODO: Generate a mask tensor of `torch.float32`s of the same shape
                # as `inputs` using either `torch.rand` or `torch.rand_like`.
                # Then replace the inputs elements whose mask value is less than
                # `self._mask_probability` with the value of `self._mask_value`.
                inputs = ...
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO: Create a word masking layer `self.MaskElements` with the given
        # `args.word_masking` probability and `MorphoDataset.UNK` as the masking value.
        self._word_masking = ...

        # TODO: Create a `torch.nn.Embedding` layer for embedding the character ids
        # from `train.forms.char_vocab` to dimensionality `args.cle_dim`.
        self._char_embedding = ...

        # TODO: Create a `torch.nn.GRU` layer processing the character embeddings,
        # producing output of dimensionality `args.cle_dim`; again, also pass
        # `batch_first=True` to the constructor.
        self._char_rnn = ...

        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the form ids
        # from `train.forms.word_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = ...

        # TODO: Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`, and you should also pass `batch_first=True` to
        # the constructor. In the forward pass, the layer will process the word embeddings
        # generated by the `self._word_embedding` layer, **now concatenated with the
        # character-level embeddings**, and we will sum the forward and backward outputs.
        self._word_rnn = ...

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.word_vocab` is the tag vocabulary.
        self._output_layer = ...

        # Initialize the layers using the Keras-inspired initialization. You can try
        # removing this line to see how much worse the default PyTorch initialization is.
        self.apply(self.keras_init)

    def forward(self, form_ids: torch.Tensor, unique_forms: torch.Tensor, form_indices: torch.Tensor) -> torch.Tensor:
        # TODO: Mask the input `form_ids` using the `self._word_masking` layer.
        hidden = ...

        # TODO: Embed the masked form IDs in `hidden` using the word embedding layer.
        hidden = ...

        # TODO: Embed the `unique_forms` using the character embedding layer.
        cle = ...

        # TODO: Pass the character embeddings through the character-level RNN.
        # As during word-level RNN, start by packing the input sequence.
        packed = ...

        # Pass the `PackedSequence` through the character RNN. Note that this time
        # we are interested only in the second output (the last hidden state of the RNN).
        _, cle = self._char_rnn(packed)

        # TODO: Concatenate the states of the forward and backward directions.
        cle = ...

        # TODO: With `cle` being the character-level embeddings of the unique forms
        # of shape `[num_unique_forms, 2 * cle_dim]`, create the representation of the
        # (not necessary unique) sentence forms by indexing the character-level
        # embeddings with the `form_indices`. The result should have a shape
        # `[batch_size, max_sentence_length, 2 * cle_dim]`. You can use for example
        # the `torch.nn.functional.embedding` function.
        cle = ...

        # TODO: Concatenate the word embeddings with the character-level embeddings (in this order).
        hidden = ...

        # TODO(tagger_we): Process the embeddings through the RNN layer. Because the sentences
        # have different length, you have to use `torch.nn.utils.rnn.pack_padded_sequence`
        # to construct a variable-length `PackedSequence` from the input. You need to compute
        # the length of each sentence in the batch (by counting non-`MorphoDataset.PAD` tokens);
        # note that these lengths must be on CPU, so you might need to use the `.cpu()` method.
        # Finally, also pass `batch_first=True` and `enforce_sorted=False` to the call.
        packed = ...

        # Pass the `PackedSequence` through the RNN.
        hidden, _ = self._word_rnn(packed)

        # TODO(tagger_we): Unpack the RNN output using the `torch.nn.utils.rnn.pad_packed_sequence` with
        # `batch_first=True` argument. Then sum the outputs of forward and backward directions.
        hidden = ...

        # TODO(tagger_we): Pass the RNN output through the output layer. Such an output has a shape
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
        # Note that compared to `tagger_we`, we also return the original
        # forms in order to be able to compute the character-level embeddings.
        return form_ids, example["forms"], tag_ids
    train = morpho.train.transform(prepare_tagging_data)
    dev = morpho.dev.transform(prepare_tagging_data)

    def prepare_batch(data):
        # Construct a single batch, where `data` is a list of examples
        # generated by `prepare_tagging_data`.
        form_ids, forms, tag_ids = zip(*data)
        # TODO(tagger_we): Combine `form_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        form_ids = ...
        # TODO: Create required inputs for the character-level embeddings using
        # the provided `morpho.train.cle_batch` function on `forms`. The function
        # returns a pair of two PyTorch tensors:
        # - `unique_forms` with shape `[num_unique_forms, max_form_length]` containing
        #   each unique form as a sequence of character ids,
        # - `forms_indices` with shape `[num_sentences, max_sentence_length]`
        #   containing for every form its index in `unique_forms`.
        unique_forms, forms_indices = ...
        # TODO(tagger_we): Process `tag_ids` analogously to `form_ids`.
        tag_ids = ...
        return (form_ids, unique_forms, forms_indices), tag_ids
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

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
