#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data
    dataset = ReadingComprehensionDataset()

    # TODO: Create the model and train it.
    model = ...

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = ...

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
