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


# The GAN model
class GAN(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        # TODO: Define `self.generator` as a `keras.Model`, which
        # - takes vectors of shape `[args.z_dim]` on input
        # - applies batch normalized dense layer with 1_024 units and ReLU
        # - applies batch normalized dense layer with `MNIST.H // 4 * MNIST.W // 4 * 64` units and ReLU
        # - reshapes the current hidden output to `[MNIST.H // 4, MNIST.W // 4, 64]`
        # - applies batch normalized transposed convolution with 32 filters, kernel size 4,
        #   stride 2, same padding, and ReLU activation
        # - applies transposed convolution with `MNIST.C` filters, kernel size 4,
        #   stride 2, same padding, and a suitable output activation
        # Note that on the lecture, we discussed that layers before batch normalization should
        # not use bias -- but for simplicity, do not do it here (so do not set `use_bias=False`).
        self.generator = ...

        # TODO: Define `self.discriminator` as a `keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation
        # - max-pools with pool size 2 and stride 2
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation
        # - max-pools with pool size 2 and stride 2
        # - flattens the current representation
        # - applies batch normalized dense layer with 1_024 units and ReLU activation
        # - applies output dense layer with one output and a suitable activation function
        self.discriminator = ...

        self.tb_callback = TorchTensorBoardCallback(args.logdir)

    # We override `compile`, because we want to use two optimizers.
    def compile(
        self, discriminator_optimizer: keras.optimizers.Optimizer, generator_optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss, metric: keras.metrics.Metric,
    ) -> None:
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss = loss
        self.metric = metric
        self.built = True

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # TODO(gan): Train the generator:
        # - generate as many random latent samples as there are `images`, by a single call
        #   to `self._z_prior.sample`;
        # - pass the samples through a generator; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
        # - compute `generator_loss` using `self.loss`, with ones as target labels
        #   (`torch.ones_like` might come handy).
        # Then, run an optimizer step with respect to generator trainable variables.
        # Do not forget that we created `generator_optimizer` in the `compile` override.

        # TODO(gan): Train the discriminator:
        # - discriminate `images` with `training=True`, storing
        #   results in `discriminated_real`
        # - discriminate images generated in generator training with `training=True`,
        #   storing results in `discriminated_fake`
        # - compute `discriminator_loss` by summing
        #   - `self.loss` on `discriminated_real` with suitable targets,
        #   - `self.loss` on `discriminated_fake` with suitable targets.
        # Then, run an optimizer step with respect to discriminator trainable variables.
        # Do not forget that we created `discriminator_optimizer` in the `compile` override.

        # TODO(gan): Update the discriminator accuracy metric -- call the
        # `self.metric` twice, with the same arguments the `self.loss`
        # was called during discriminator loss computation.

        self._loss_tracker.update_state(discriminator_loss + generator_loss)
        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            **self.get_metrics_result(),
        }

    def generate(self, epoch: int, logs: dict[str, torch.Tensor]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._z_prior.sample([GRID * GRID]), training=False)

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
        interpolated_images = self.generator(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = torch.cat([
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(random_images, GRID)], axis=0),
            torch.zeros([MNIST.H * GRID, MNIST.W, MNIST.C]),
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(interpolated_images, GRID)], axis=0),
        ], axis=1)
        self.tb_callback.writer("train").add_image("images", image, epoch + 1, dataformats="HWC")


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

    # Load data
    mnist = MNIST(args.dataset, size={"train": args.train_size})
    train = mnist.train.transform(lambda example: example["image"] / 255)
    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # Create the network and train
    network = GAN(args)
    network.compile(
        discriminator_optimizer=keras.optimizers.Adam(),
        generator_optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metric=keras.metrics.BinaryAccuracy("discriminator_accuracy"),
    )
    logs = network.fit(train, epochs=args.epochs, callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=network.generate), network.tb_callback])

    # Return the loss and the discriminator accuracy for ReCodEx to validate.
    return {metric: logs.history[metric][-1] for metric in ["loss", "discriminator_accuracy"]}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
