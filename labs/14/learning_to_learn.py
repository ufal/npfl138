#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Iterable
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from omniglot_dataset import Omniglot

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--cell_size", default=40, type=int, help="Memory cell size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes per episode.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--images_per_class", default=10, type=int, help="Images per class.")
parser.add_argument("--lstm_dim", default=256, type=int, help="LSTM Dim")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--read_heads", default=1, type=int, help="Read heads.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--test_episodes", default=1_000, type=int, help="Number of testing episodes.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_episodes", default=10_000, type=int, help="Number of training episodes.")
# If you add more arguments, ReCodEx will keep them with your default values.


class EpisodeGenerator(torch.utils.data.IterableDataset):
    """Python generator of episodes."""
    def __init__(self, dataset: Omniglot.Dataset, size: int, args: argparse.Namespace, seed: int) -> None:
        self._dataset = dataset
        self._size = size
        self._args = args

        # Random generator
        self._generator = np.random.RandomState(seed)

        # Create required indexes
        self._unique_labels = np.unique(dataset.data["labels"])
        self._label_indices = {}
        for i, label in enumerate(dataset.data["labels"]):
            self._label_indices.setdefault(label, []).append(i)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterable[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
        """Generate infinite number of episodes.

        Every episode contains `self._args.classes` randomly chosen Omniglot
        classes, each class being assigned a randomly chosen label. For every
        chosen class, `self._args.images_per_class` images are randomly selected.

        Apart from the images, the input contains the random labels one step
        after the corresponding images (with the first label being -1).
        The gold outputs are also the labels, but without the one-step offset.
        """
        for _ in range(self._size):
            indices, labels = [], []
            for index, label in enumerate(self._generator.choice(
                    self._unique_labels, size=self._args.classes, replace=False)):
                indices.extend(self._generator.choice(
                    self._label_indices[label], size=self._args.images_per_class, replace=False))
                labels.extend([index] * self._args.images_per_class)
            indices, labels = np.array(indices, np.int32), np.array(labels, np.int32)

            permutation = self._generator.permutation(len(indices))
            images = self._dataset.data["images"][indices[permutation]]
            labels = labels[permutation]
            yield (images, np.pad(labels[:-1], [[1, 0]], constant_values=-1)), labels


class Model(keras.Model):
    class NthOccurenceAccuracy(keras.metrics.SparseCategoricalAccuracy):
        """A sparse categorical accuracy computed only for `nth` occurrence of every element."""
        def __init__(self, nth: int, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._nth = nth

        def update_state(self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None) -> None:
            assert sample_weight is None
            y_true = keras.ops.convert_to_tensor(y_true)
            one_hot = keras.ops.one_hot(y_true, keras.ops.max(y_true) + 1)
            nth = keras.ops.sum(keras.ops.cumsum(one_hot, axis=-2) * one_hot, axis=-1)
            indices = torch.where(nth == self._nth)
            return super().update_state(y_true[*indices], y_pred[*indices])

    class MemoryAugmentedLSTMCell(keras.layers.Layer):
        """The LSTM controller augmented with external memory.

        The LSTM has dimensionality `units`. The external memory consists
        of `memory_cells` cells, each being a vector of `cell_size` elements.
        The controller has `read_heads` read heads and one write head.
        """
        def __init__(self, units: int, memory_cells: int, cell_size: int, read_heads: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self._memory_cells = memory_cells
            self._cell_size = cell_size
            self._read_heads = read_heads
            self.built = True

            # TODO: Create the required layers:
            # - `self._controller` is a `keras.layers.LSTMCell` with `units` units;
            # - `self._parameters_layer` is a `tanh`-activated dense layer with `(read_heads + 1) * cell_size` units;
            # - `self._output_layer` is a `tanh`-activated dense layer with `units` units.
            self._controller = ...
            self._parameters_layer = ...
            self._output_layer = ...

        @property
        def output_size(self) -> int:
            # TODO: Return the size of the output of the `MemoryAugmentedLSTMCell`.
            raise NotImplementedError()

        @property
        def state_size(self) -> list[int]:
            # TODO: Assuming a state of a cell is a list of vectors for every batch example,
            # return the state sizes of the `MemoryAugmentedLSTMCell`. Notably, the state
            # consists of the following components:
            # - the external memory, a matrix containing `self._memory_cells` cells
            #   as rows, each of length `self._cell_size`, flattened into a vector;
            # - the values of memory cells read by the `self._read_heads` heads
            #   in the previous time step, flattened into a vector;
            # - all state vectors of the `self._controller` itself; note that
            #   the `self._controller` also has `state_size` property.
            raise NotImplementedError()

        def call(self, inputs: torch.Tensor, states: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
            # TODO: Decompose `states` into `memory`, `read_value`, and `controller_state`
            # (see `state_size` describing the `states` structure), and reshape `memory`
            # into a matrix.
            ...

            # TODO: Call the LSTM controller, using a concatenation of `inputs` and
            # `read_value` (in this order) as input and `controller_state` as state.
            # Store the results in `controller_output` and `controller_state`.
            controller_output, controller_state = ...

            # TODO: Pass the `controller_output` through the `self._parameters_layer`, obtaining
            # the parameters for interacting with the external memory (in this order):
            # - `write_value` is the first `self._cell_size` elements of every batch example;
            # - `read_keys` is the rest of the elements of every batch example, reshaped to
            #   `[batch_size, self._read_heads, self._cell_size]`.
            write_value = ...
            read_keys = ...

            # TODO: Read the memory. For every predicted read key, the goal is to
            # - compute cosine similarities between the key and all memory cells;
            # - compute cell distribution as a softmax of the computed cosine similarities;
            # - the read value is the sum of the memory cells weighted by the above distribution.
            #
            # However, implement the reading process in a vectorized way (for all read keys in parallel):
            # - compute L2 normalized copy of `memory` and `read_keys`, using `keras.ops.normalize`,
            #   so that every cell vector has norm 1;
            # - compute the self-attention between the L2-normalized copy of `memory` and `read_keys`
            #   with a single matrix multiplication, obtaining a tensor with shape
            #   `[batch_size, self._read_heads, self._memory_cells]`. Note that you will need
            #   to transpose one of the matrices.
            # - apply softmax, resulting in a distribution over the memory cells for every read key
            # - compute weighted sum of the original (non-L2-normalized) `memory` according to the
            #   obtained distribution. Compute it using a single matrix multiplication, producing
            #   a value with shape `[batch_size, self._read_heads, self._cell_size]`.
            # Finally, reshape the result into `read_value` of shape `[batch_size, self._read_heads * self._cell_size]`
            read_value = ...

            # TODO: Write to the memory by prepending the `write_value` as the first cell (row);
            # the last memory cell (row) is dropped.
            memory = ...

            # TODO: Generate `output` by concatenating `controller_output` and `read_value`
            # (in this order) and passing it through the `self._output_layer`.
            output = ...

            # TODO: Return the `output` as output and `memory`, `read_value`, and all vectors of
            # `constroller_state` as the state.
            raise NotImplementedError()

    def __init__(self, args: argparse.Namespace) -> None:
        # Construct the model. The inputs are:
        # - a sequence of `images`;
        # - a sequence of labels of the previous images.
        images = keras.Input(
            [args.classes * args.images_per_class, Omniglot.H, Omniglot.W, Omniglot.C], dtype="float32")
        previous_labels = keras.Input([args.classes * args.images_per_class], dtype="int32")

        # TODO: Process each image with the same sequence of the following operations:
        # - start by reshaping the `images` so that they are a batch of individual images,
        #   not a batch of sequences of images;
        # - apply the `keras.layers.Rescaling(1/255.)` layer to scale the images to [0, 1] range;
        # - convolutional layer with 8 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 16 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 32 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - flatten each image into a vector;
        # - finally, reshape the batch back into a sequence of image representations.
        # As in the `gan` assignment, for simplicity do not pass the `use_bias=False` argument.

        # TODO: To create the input for the `MemoryAugmentedLSTMCell`, concatenate (in this order)
        # each computed image representation with the one-hot representation (with `args.classes` classes)
        # of the label of the previous image from `previous_labels`.

        # TODO: Create the `MemoryAugmentedLSTMCell` cell, using
        # - `args.lstm_dim` units;
        # - `args.classes * args.images_per_class` memory cells of size `args.cell_size`;
        # - `args.read_heads` read heads.
        # Then, run this cell using `keras.layers.RNN` on the prepared input,
        # obtaining output for every input sequence element (see `return_sequences` argument).

        # TODO: Pass the sequence of outputs through a classification dense layer
        # with `args.classes` units and `"softmax"` activation.
        predictions = ...

        # Create the model and compile it.
        super().__init__(inputs=[images, previous_labels], outputs=predictions)
        self.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[*[self.NthOccurenceAccuracy(i, name="acc{:02d}".format(i)) for i in [1, 2, 5, 10]],
                     keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The images contain a single channel of values
    # of type `torch.uint8` in [0-255] range.
    omniglot = Omniglot()

    train = EpisodeGenerator(omniglot.train, args.train_episodes, args, seed=args.seed)
    test = list(EpisodeGenerator(omniglot.test, args.test_episodes, args, seed=42))

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

    # Create the model and train
    model = Model(args)
    logs = model.fit(train, epochs=args.epochs, validation_data=test)

    # Return the training and development losses for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
