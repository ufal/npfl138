#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.2")
from npfl138 import GymCartpoleDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.pt", type=str, help="Output model path.")


def evaluate_model(
    model: torch.nn.Module, seed: int = 42, episodes: int = 100, render: bool = False, report_per_episode: bool = False
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            prediction = model(torch.from_numpy(observation)).numpy(force=True)
            assert len(prediction) == 2, "The model must output two values."
            action = np.argmax(prediction)

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        # TODO: Create the model layers, with the last layer having 2 outputs.
        # To store a list of layers, you can use either `torch.nn.Sequential`
        # or `torch.nn.ModuleList`; you should *not* use a Python list.
        ...

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Run your model. Because some inputs are on a CPU, you should
        # start by moving them to the `model.device`.
        ...


def main(args: argparse.Namespace) -> torch.nn.Module | None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    if not args.evaluate:
        if args.batch_size is ...:
            raise ValueError("You must specify the batch size, either in the defaults or on the command line.")
        if args.epochs is ...:
            raise ValueError("You must specify the number of epochs, either in the defaults or on the command line.")

        # Create logdir name.
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ))

        # Load the provided dataset. The `dataset.train` is a collection of 100 examples,
        # each being a pair of (inputs, label), where:
        # - `inputs` is a vector with `GymCartpoleDataset.FEATURES` floating point values,
        # - `label` is a gold 0/1 class index.
        dataset = GymCartpoleDataset()

        train = torch.utils.data.DataLoader(dataset.train, args.batch_size, shuffle=True)

        model = Model(args)

        # TODO: Configure the model for training.
        model.configure(...)

        # TODO: Train the model. Note that you can pass a list of callbacks to the
        # `fit` method, each being a callable accepting the model, epoch, and logs.
        # Such callbacks are called after every epoch and if they modify the
        # logs dictionary, the values are logged on the console and to TensorBoard.
        model.fit(train, epochs=args.epochs, callbacks=[])

        # Save the model, both the hyperparameters and the parameters. If you
        # added additional arguments to the `Model` constructor beyond `args`,
        # you would have to add them to the `save_config` call below.
        model.save_config(f"{args.model}.json", args=args)
        model.save_weights(args.model)

    else:
        # Evaluating, either manually or in ReCodEx.
        model = Model(**Model.load_config(f"{args.model}.json"))
        model.load_weights(args.model)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print("The average score was {}.".format(score))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
