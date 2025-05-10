#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.12")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        return image, image  # return the image both as the input and the target


class GAN(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = lambda: torch.distributions.Normal(  # Lambda method to construct the
            torch.zeros(args.z_dim, device=self.device),     # prior distribution on the current
            torch.ones(args.z_dim, device=self.device))      # device of the model.

        # TODO: Define `self.generator` as a `torch.nn.Sequential` module, which
        # - takes vectors of `[args.z_dim]` shape on input;
        # - applies batch normalized dense layer with 1_024 units and ReLU;
        # - applies batch normalized dense layer with `64 * MNIST.H // 4 * MNIST.W // 4` units and ReLU;
        # - reshapes the current hidden output to `[64, MNIST.H // 4, MNIST.W // 4]`;
        # - applies batch normalized transposed convolution with 32 filters, kernel size 4,
        #   stride 2, padding 1, and ReLU activation;
        # - applies transposed convolution with `MNIST.C` filters, kernel size 4,
        #   stride 2, padding 1, and a suitable output activation.
        # You can use lazy layer or regular layers, but you should use them consistently;
        # so either all your layers should be lazy, or all of them should be regular.
        #
        # Pass epsilon of 0.01 to all batch normalizations for better stability.
        # Moreover, on the lecture we discussed that the layers before batch normalization
        # should not use bias, so do not forget to set `bias=False` on all such layers.
        self.generator = ...

        # TODO: Define `self.discriminator` as a `torch.nn.Sequential`, which
        # - takes input images with shape `[MNIST.C, MNIST.H, MNIST.W]`;
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation;
        # - max-pools with pool size 2 and stride 2;
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation;
        # - max-pools with pool size 2 and stride 2;
        # - flattens the current representation;
        # - applies batch normalized dense layer with 1_024 units and ReLU activation
        # - applies output dense layer with one output and a suitable activation function.
        #
        # Again, pass epsilon of 0.01 to all batchnorms and use `bias=False` where appropriate.
        self.discriminator = ...

    def train_step(self, xs: tuple[torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        images = xs[0]

        # TODO(gan): Train the generator:
        # - generate as many random latent samples as there are `images`, by a single call
        #   to `self._z_prior.sample`;
        # - pass the samples through the generator;
        # - run discriminator on the generated images (keep it running in the training mode,
        #   even if not updating its parameters, we want to perform possible BatchNorm in it);
        # - compute `generator_loss` using `self.loss`, with ones as target labels.
        # Then, perform a step of the generator optimizer stored in `self.optimizer["generator"]`.
        ...

        # TODO(gan): Train the discriminator:
        # - first run the discriminator on `images`, storing the results in `discriminated_real`;
        # - then process the images generated during the generator training, storing the results
        #   in `discriminated_fake` (be careful to neither re-run the generator nor perform
        #   backpropagation into the generator during the discriminator loss computation);
        # - compute `discriminator_loss` by summing:
        #   - `self.loss` on `discriminated_real` with suitable targets,
        #   - `self.loss` on `discriminated_fake` with suitable targets.
        # Then, perform a step of the discriminator optimizer stored in `self.optimizer["discriminator"]`.
        ...

        # TODO(gan): Update the discriminator accuracy metric -- call the
        # `self.metrics["discriminator_accuracy"].update` twice, with the same
        # arguments the `self.loss` was called during discriminator loss computation.
        ...

        # Return the mean of the overall loss, the current discriminator and generator losses, and the metrics.
        loss = self.loss_tracker(discriminator_loss + generator_loss)
        return {"loss": loss, "discriminator_loss": discriminator_loss, "generator_loss": generator_loss,
                **{metric: self.metrics[metric].compute() for metric in self.metrics}}

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20

        self.generator.eval()
        with torch.no_grad(), torch.device(self.device):
            # Generate GRIDxGRID images.
            random_images = self.generator(self._z_prior().sample([GRID * GRID]))

            # Generate GRIDxGRID interpolated images.
            if self._z_dim == 2:
                # Use 2D grid of Z values for interpolated images.
                starts = torch.stack([-2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
                ends = torch.stack([2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
            else:
                # Otherwise generate random Z for the first and the last column.
                starts, ends = self._z_prior().sample([2, GRID])
            interpolated_z = torch.cat(
                [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID).unsqueeze(-1) for i in range(GRID)])
            interpolated_images = self.generator(interpolated_z)

            # Stack the random images, then an empty column, and finally interpolated images.
            grid = torch.cat([
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(random_images, GRID)], dim=1),
                torch.zeros([MNIST.C, MNIST.H * GRID, MNIST.W]),
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(interpolated_images, GRID)], dim=1),
            ], dim=2)
            self.get_tb_writer("train").add_image("images", grid, epoch)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST(args.dataset, sizes={"train": args.train_size})
    train = TrainableDataset(mnist.train).dataloader(args.batch_size, shuffle=True, seed=args.seed)

    # Create the model and train it.
    model = GAN(args)

    # TODO(gan): Create Adam optimizers for the generator and the discriminator,
    # and the loss function and metric for the discriminator.
    model.configure(
        optimizer={
            "discriminator": ...,
            "generator": ...,
        },
        loss=...,
        metrics={"discriminator_accuracy": ...},
        logdir=args.logdir,
    )

    logs = model.fit(train, epochs=args.epochs, callbacks=[GAN.generate])

    # Return the loss and the discriminator accuracy for ReCodEx to validate.
    return {metric: logs[metric] for metric in ["train_loss", "train_discriminator_accuracy"]}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
