#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

from trainable_module import TrainableModule
from text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(TrainableModule):
    def __init__(self, args, eleczech, dataset):
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        ...

        # In the previous assignments, we often used
        #   self.apply(self.keras_init)
        # If you use it here, be careful not to overwrite the already pre-trained `eleczech` model.

    # TODO: Implement the model computation.
    def forward(self, ...) -> torch.Tensor:
        raise NotImplementedError()


def main(args: argparse.Namespace) -> None:
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

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # Create the dataloaders
    def prepare_example(example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        raise NotImplementedError()

    def prepare_batch(data):
        # TODO: Construct a single batch using a list of examples from the `prepare_example` function.
        raise NotImplementedError()

    def create_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset.transform(prepare_example), args.batch_size, shuffle, collate_fn=prepare_batch)
    train = create_dataloader(facebook.train, shuffle=True)
    dev = create_dataloader(facebook.dev, shuffle=False)
    test = create_dataloader(facebook.test, shuffle=False)

    # Create the model
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    model.configure(
        optimizer=...,
        loss=...,
        metrics=...,
        ...,
    )
    model.fit(...)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = ...

        for sentence in predictions:
            print(facebook.train.label_vocab.string(np.argmax(sentence)), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
