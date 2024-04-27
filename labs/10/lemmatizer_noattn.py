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
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.forms.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO: Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = ...
        self._source_rnn = ...

        # TODO: Then define
        # - `self._target_rnn` as a unidirectional GRU layer with `args.rnn_dim` units processing
        #   embedded target characters
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_rnn = ...
        self._target_output_layer = ...

        # Create self._target_rnn_cell, which is the single cell of `self._target_rnn`.
        self._target_rnn_cell = torch.nn.GRUCell(args.rnn_dim, args.rnn_dim)
        for name, _ in self._target_rnn_cell.named_parameters():
            setattr(self._target_rnn_cell, name, getattr(self._target_rnn, "{}_l0".format(name)))

        if not args.tie_embeddings:
            # TODO: Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = ...
        else:
            # TODO: Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            self._target_embedding = ...

        # Initialize the layers using the Keras-inspired initialization. You can try
        # removing this line to see how much worse the default PyTorch initialization is.
        self.apply(self.keras_init)

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, forms: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(forms)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=encoded.shape[1] + 10)

    def encoder(self, forms: torch.Tensor) -> torch.Tensor:
        # TODO: Embed the inputs using `self._source_embedding`.

        # TODO: Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. The result should be the last hidden states of the forward and
        # backward direction, summed together.
        return ...

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - the `self._target_rnn` layer, passing an additional parameter `initial_state=[encoded]`,
        #   and again correctly handling padding,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.
        return ...

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = ...
        inputs = ...
        states = ...
        results = ...
        result_lengths = ...

        while index < max_length and torch.any(result_lengths == max_length):
            # TODO:
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn.cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            predictions = ...

            # Store the predictions in the `result` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # TODO: Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = ...
            index = ...

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs, training):
        if training:
            y_pred = y_pred.argmax(dim=-2)
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"](torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return self.metrics.compute()

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if self._batches % self._show_results_every_batch == 0:
            self._tqdm.write("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(np.trim_zeros(xs[0][0].numpy(force=True)))),
                "".join(self._target_vocab.strings(self.predict_step((xs[0][:1],))[0]))))

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs, training=False)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # If `as_numpy==True`, trim the predictions at the first EOW.
            if as_numpy:
                batch = [example[np.cumsum(example == MorphoDataset.EOW) == 0] for example in batch.numpy(force=True)]
            return batch


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
        # TODO: Return `example["forms"]` as inputs and `example["lemmas"]` as targets.
        raise NotImplementedError()
    train = morpho.train.transform(prepare_tagging_data)
    dev = morpho.dev.transform(prepare_tagging_data)

    def prepare_batch(data, training: bool):
        forms, lemmas = zip(*data)
        # TODO: The `forms` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `morpho.train.forms.char_vocab` vocabulary.
        # Then create a tensor by padding the forms to the length of the longest one in the batch.
        forms = ...
        # TODO: Process `lemmas` analogously to `forms`, but use `morpho.train.lemmas.char_vocab`,
        # and additionally, append `MorphoDataset.EOW` to the end of each lemma.
        lemmas = ...
        # In the training regime, we pass `lemmas` also as inputs.
        return ((forms, lemmas) if training else forms), lemmas
    def prepare_training_batch(data):
        return prepare_batch(data, training=True)
    def prepare_dev_batch(data):
        return prepare_batch(data, training=False)
    train = torch.utils.data.DataLoader(train, args.batch_size, collate_fn=prepare_training_batch, shuffle=True)
    dev = torch.utils.data.DataLoader(dev, args.batch_size, collate_fn=prepare_dev_batch)

    model.configure(
        # TODO: Create the optimizer by creating an instance of
        # `torch.optim.Adam` which will train the `model.parameters()`.
        optimizer=...,
        # TODO: Use `torch.nn.CrossEntropyLoss` to instantiate the loss function.
        # Pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=...,
        # TODO: Create a `torchmetrics.MeanMetric()` metric, where we will manually
        # collect lemmatization accuracy.
        metrics={"accuracy": ...},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs, verbose=2 if args.show_results_every_batch else 1)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
