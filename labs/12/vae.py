#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from mnist_v2 import MNIST

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


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


# The VAE model
class VAE(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.built = True

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        # TODO: Using functional API, define `self.encoder` as a `keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - flattens them
        # - applies `len(args.encoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.encoder_layers[i]` units
        # - generates two outputs `z_mean` and `z_sd`, each passing the result
        #   of the above bullet through its own dense layer of `args.z_dim` units,
        #   with `z_sd` using exponential function as activation to keep it positive.
        self.encoder = ...

        # TODO: Using functional API, define `self.decoder` as a `keras.Model`, which
        # - takes vectors of `[args.z_dim]` shape on input
        # - applies `len(args.decoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.decoder_layers[i]` units
        # - applies output dense layer with `MNIST.H * MNIST.W * MNIST.C` units
        #   and a suitable output activation
        # - reshapes the output (`keras.layers.Reshape`) to `[MNIST.H, MNIST.W, MNIST.C]`
        self.decoder = ...

        self.tb_callback = TorchTensorBoardCallback(args.logdir)

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.zero_grad()

        # TODO: Compute `z_mean` and `z_sd` of the given images using `self.encoder`.
        # Note that you should pass `training=True` to the `self.encoder`.

        # TODO: Sample `z` from a Normal distribution with mean `z_mean` and
        # standard deviation `z_sd`. Start by creating corresponding
        # distribution `torch.distributions.Normal(...)` and then run the
        # `rsample()` method. The `rsample()` method performs sampling using
        # the reparametrization trick, or fails when it is not supported
        # by the distribution.

        # TODO: Decode images using `z` (also passing `training=True` to the `self.decoder`).

        # TODO: Compute `reconstruction_loss` using `self.compute_loss(x, y_true, y_pred)`.
        reconstruction_loss = ...

        # TODO: Compute `latent_loss` as a mean of KL divergences of suitable distributions.
        # Note that PyTorch offers `torch.distributions.kl.kl_divergence` computing
        # the exact KL divergence of two given distributions.
        latent_loss = ...

        # TODO: Compute `loss` as a sum of the `reconstruction_loss` (multiplied by the number
        # of pixels in an image) and the `latent_loss` (multiplied by self._z_dim).
        loss = ...

        # TODO: Perform a single step of the `self.optimizer`, with respect to `self.trainable_variables`,
        # which are trainable variables of both the encoder and the decoder.
        ...

        self._loss_tracker.update_state(loss)
        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.decoder(self._z_prior.sample([GRID * GRID]), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = torch.stack([-2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
            ends = torch.stack([2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts = self._z_prior.sample([GRID])
            ends = self._z_prior.sample([GRID])
        interpolated_z = torch.cat(
            [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID).unsqueeze(-1) for i in range(GRID)])
        interpolated_images = self.decoder(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = torch.cat([
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(random_images, GRID)], axis=0),
            torch.zeros([MNIST.H * GRID, MNIST.W, MNIST.C]),
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(interpolated_images, GRID)], axis=0),
        ], axis=1)
        self.tb_callback.writer("train").add_image("images", image, epoch + 1, dataformats="HWC")


def main(args: argparse.Namespace) -> float:
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

    # Load data
    mnist = MNIST(args.dataset, size={"train": args.train_size})
    train = mnist.train.transform(lambda example: example["image"] / 255)
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # Create the network and train
    network = VAE(args)
    network.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy())
    logs = network.fit(train, epochs=args.epochs, callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=network.generate), network.tb_callback])

    # Return loss for ReCodEx to validate
    return logs.history["loss"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
