#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        ...

    # TODO: Implement the model computation.
    def forward(self, *REPLACE_WITH_YOUR_NETWORK_INPUTS) -> torch.Tensor:
        raise NotImplementedError()


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, *MAYBE_ADD_ARGUMENTS) -> None:
        super().__init__(dataset)

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        raise NotImplementedError()

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        raise NotImplementedError()


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    ...

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    model.configure(
        optimizer=...,  # maybe also a scheduler, but not required
        loss=...,
        metrics=...,
        logdir=logdir,
    )
    model.fit(...)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(...)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(document_logits.argmax().item()), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
