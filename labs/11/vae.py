#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers.")
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


class VAE(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = lambda: torch.distributions.Normal(  # Lambda method to construct the
            torch.zeros(args.z_dim, device=self.device),     # prior distribution on the current
            torch.ones(args.z_dim, device=self.device))      # device of the model.

        # TODO: Define `self.encoder` as a `torch.nn.Sequential` module, which
        # - takes input images with shape `[MNIST.C, MNIST.H, MNIST.W]`;
        # - flattens them;
        # - applies `len(args.encoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.encoder_layers[i]` units;
        # - generates an output of shape `[2 * args.z_dim]` by applying an output
        #   linear layer. During training, this output will be split into two,
        #   the `z_mean` and then the logarithm of `z_sd`.
        # You can use lazy layer or regular layers, but you should use them consistently;
        # so either all your layers should be lazy, or all of them should be regular.
        self.encoder = ...

        # TODO: Define `self.decoder` as a `torch.nn.Sequential`, which
        # - takes vectors of `[args.z_dim]` shape on input;
        # - applies `len(args.decoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.decoder_layers[i]` units;
        # - applies output dense layer with `MNIST.C * MNIST.H * MNIST.W` units
        #   and sigmoid activation;
        # - uses `torch.nn.Unflatten` to reshape the output to `[MNIST.C, MNIST.H, MNIST.W]`.
        self.decoder = ...

    def train_step(self, xs: tuple[torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        images = xs[0]

        # TODO: Compute `z_mean` and `z_sd` of the given images using `self.encoder`.
        # The `z_mean` is the first half of the output of the encoder; the `z_sd`
        # is the second half of the output of the encoder passed through `torch.exp`.

        # TODO: Sample `z` from a Normal distribution with mean `z_mean` and
        # standard deviation `z_sd`. Start by creating corresponding
        # distribution `torch.distributions.Normal(...)` and then run the
        # `rsample()` method. The `rsample()` method performs sampling using
        # the reparametrization trick, or fails when it is not supported
        # by the distribution.

        # TODO: Decode images using the sampled `z`.

        # TODO: Compute `reconstruction_loss` using an appropriate loss from `torch.nn.functional`.
        reconstruction_loss = ...

        # TODO: Compute `latent_loss` as a mean of KL divergences of suitable distributions.
        # Note that PyTorch offers `torch.distributions.kl.kl_divergence` computing
        # the exact KL divergence of two given distributions.
        latent_loss = ...

        # TODO: Compute `loss` as a sum of the `reconstruction_loss` (multiplied by the number
        # of pixels in an image) and the `latent_loss` (multiplied by self._z_dim).
        loss = ...

        # TODO: Perform a single step of the `self.optimizer` (both encoder and
        # decoder parameters should be updated).
        ...

        # Return the mean of the overall loss, and the current reconstruction and latent losses.
        loss = self.loss_tracker(loss)
        return {"loss": loss, "reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss}

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20

        self.decoder.eval()
        with torch.no_grad(), torch.device(self.device):
            # Generate GRIDxGRID images.
            random_images = self.decoder(self._z_prior().sample([GRID * GRID]))

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
            interpolated_images = self.decoder(interpolated_z)

            # Stack the random images, then an empty column, and finally interpolated images.
            grid = torch.cat([
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(random_images, GRID)], dim=1),
                torch.zeros([MNIST.C, MNIST.H * GRID, MNIST.W]),
                torch.cat([torch.cat(list(row), dim=2) for row in torch.chunk(interpolated_images, GRID)], dim=1),
            ], dim=2)
            self.get_tb_writer("train").add_image("images", grid, epoch)


def main(args: argparse.Namespace) -> float:
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
    model = VAE(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        logdir=args.logdir,
    )

    logs = model.fit(train, epochs=args.epochs, callbacks=[VAE.generate])

    # Return the training loss for ReCodEx to validate.
    return logs["train_loss"]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
