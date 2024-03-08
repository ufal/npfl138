#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create models
    models = []
    for model in range(args.models):
        models.append(keras.Sequential([
            keras.layers.Rescaling(1 / 255),
            keras.layers.Flatten(),
        ] + [keras.layers.Dense(hidden_layer, activation="relu") for hidden_layer in args.hidden_layers] + [
            keras.layers.Dense(MNIST.LABELS, activation="softmax"),
        ]))

        models[-1].compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model + 1), end="", flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done")

    individual_accuracies, ensemble_accuracies = [], []
    for model in range(args.models):
        # TODO: Compute the accuracy on the dev set for the individual `models[model]`.
        individual_accuracy = ...

        # TODO: Compute the accuracy on the dev set for the ensemble `models[0:model+1]`.
        #
        # Generally you can choose one of the following approaches:
        # 1) Use Keras Functional API and construct a `keras.Model` averaging the models
        #    in the ensemble (using for example `keras.layers.Average` or manually
        #    with `keras.ops.mean`). Then you can compile the model with the required metric
        #    (and with a loss; but an optimizer is not required) and use `model.evaluate`.
        # 2) Manually perform the averaging (using PyTorch or NumPy). In this case you do not
        #    need to construct Keras ensemble model at all, and instead call `model.predict`
        #    on the individual models and average the results. To measure accuracy,
        #    either do it completely manually or use `keras.metrics.SparseCategoricalAccuracy`.
        ensemble_accuracy = ...

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
