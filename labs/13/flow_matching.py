#!/usr/bin/env python3
import argparse
import copy
import datetime
import os
import re

import torch

import npfl138
npfl138.require_version("2425.13")
from npfl138.datasets.image64_dataset import Image64Dataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="L1Loss", type=str, help="The loss to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--sigma_min", default=0.001, type=float, help="Sigma_min used in OT paths.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The diffusion model architecture building blocks.
class SinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embeddings used to embed the current time step."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == 1

        # TODO: Compute the sinusoidal embeddings of the inputs in `[0, 1]` range.
        # The `inputs` have shape `[..., 1]`, and the produced embeddings should have
        # shape `[..., self.dim]`, where for `0 <= i < self.dim/2`,
        # - the value on index `[..., i]` should be
        #     `sin(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        # - the value on index `[..., self.dim/2 + i]` should be
        #     `cos(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        raise NotImplementedError()


class ResidualBlock(torch.nn.Module):
    """A residual block with two 3x3 convolutions and a time embedding."""
    def __init__(self, width: int) -> None:
        super().__init__()
        # TODO: In the whole assignment, use _lazy_ convolutional and _lazy_
        # linear layers. Furthermore, create the layers in the exact same order
        # as mentioned in the comment.
        #
        # Create the layers of the residual block, which works as follows:
        # - The input images are passed through a 3x3 convolution with
        #   `width` channels and "same" padding.
        # - Then, the outputs are passed through a group normalization layer
        #   with `width` channels and `min(width // 4, 16)` groups.
        # - The result is passed through the swish activation function.
        # - Then, the time embeddings are passed through a linear layer with
        #   `width` outputs and swish activation, and then added to the
        #   convolutional features from the previous step.
        # - Finally, the result is passed through another 3x3 convolution
        #   convolution with `width` channels and "same" padding, and a group
        #   normalization with the same number of channels and groups as before.
        # - The result of the block is then added to the input images and returned.
        #
        # During initialization, set the `weight` parameter of the last GroupNorm
        # layer to all zeros (so that the block produces initially only zeros).
        #
        # As mentined earlier, every convolutional layer before a GroupNorm layer
        # must not have a bias (it is provided by the group normalization).
        ...

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the residual block. The `times` has
        # shape `[batch_size, channels]` with `channels` equal to the number of
        # `images` channels, so it must be broadcasted to every image position.
        raise NotImplementedError()


class DownscalingBlock(torch.nn.Module):
    """Downscaling block returning both the features of original and downscaled size."""
    def __init__(self, residual_blocks: int, width: int) -> None:
        super().__init__()
        # TODO: The downscaling block starts with `residual_blocks` number of `ResidualBlock`s,
        # and then passes the result through a 3x3 convolution with `width << 1` channels,
        # stride 2, and padding 1.
        ...

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement the forward pass of the downscaling block, returning a pair with
        # first the downscaled features followed by the output of the last residual block.
        raise NotImplementedError()


class UpscalingBlock(torch.nn.Module):
    """Upscaling block using a skip connection from the corresponding downscaling block."""
    def __init__(self, residual_blocks: int, width: int) -> None:
        super().__init__()
        # TODO: The upscaling block starts with a transposed convolution with kernel size 4,
        # stride 2, and padding 1, which processes the input images. Then, the skip connection
        # from the downscaling block is passed through a 3x3 convolution with `width` channels
        # and the "same" padding. Finally, both results are added together and passed through
        # `residual_blocks` number of `ResidualBlock`s.
        ...

    def forward(self, images: torch.Tensor, skip_connections: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass of the upscaling block.
        raise NotImplementedError()


class UNet(torch.nn.Module):
    """The U-Net architecture used in the flow matching model."""
    def __init__(self, channels: int, stage_blocks: int, stages: int) -> None:
        super().__init__()
        # TODO: When processing the input images and the input times, start by
        # passing the times through the `SinusoidalEmbedding` layer; the result
        # is then passed to all later layers that require the time embedding.
        #
        # The U-Net architecture consists of the following layers:
        # - the initial 3x3 convolution with `channels` channels, the "same"
        #   padding, and no activation;
        # - then `stages` number of `DownscalingBlock`s, each having `stage_blocks`
        #   residual blocks. The first residual block has `channels` channels, and
        #   every next block has twice as many channels;
        # - then a middle block composed of `stage_blocks` number of `ResidualBlock`s
        #   with `channels << stages` channels;
        # - then `stages` number of `UpscalingBlock`s, each having `stage_blocks` of
        #   `ResidualBlock`s. The first upscaling block has the same number of channels
        #   as the last downscaling block, and every following upscaling block has half
        #   the channels as the previous one, and processes also the skip connection
        #   from the corresponding downscaling block;
        # - finally, the result is passed through an output 3x3 convolution with
        #   `C` channels and the "same" padding.
        ...

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> None:
        # TODO: Implement the forward pass of the U-Net.
        raise NotImplementedError()


class FlowMatching(npfl138.TrainableModule):
    """The model used for flow matching, capable of generating images."""
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Create the U-Net model with the required arguments.
        self._model = UNet(...)

        self._ema_model = None  # We initialize the `self._ema_model` during the first update.
        self._ema_momentum = args.ema
        self._sigma_min = args.sigma_min
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]))

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Method to normalize the input image to have a standard distribution."""
        image = (image - self.imagenet_mean[None, :, None, None]) / self.imagenet_std[None, :, None, None]
        return image

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """The inverse of the `normalize_image` method."""
        image = image * self.imagenet_std[None, :, None, None] + self.imagenet_mean[None, :, None, None]
        return image

    def train_step(self, xs: tuple[torch.tensor], y: torch.tensor) -> dict[str, torch.tensor]:
        """perform a single training update."""
        # Unpack the input batch.
        images = xs[0]

        # Generate random noise and random time steps.
        noises = torch.randn_like(images)
        times = torch.rand(images.shape[0], 1, device=images.device)

        # TODO: Perform a training step.
        # - Start by normalizing the input images using the `normalize_image` method.
        # - Then compute the noisy images used as the model input.
        # - Follow by running the model to get the predicted vector field.
        # - Finally, compute the loss using the conditional flow matching objective,
        #   utilizing the given PyTorch loss stored in `self.loss`.
        # Once the forward pass is completed, compute the gradient of the loss.
        ...

        with torch.no_grad():
            self.optimizer.step()

            # TODO: If the EMA model is not initialized, create it as a copy of the current model
            # using `copy.deepcopy`. Also call `requires_grad_(False)` on the newly created model.
            if self._ema_model is None:
                self._ema_model = ...
            for ema_variable, variable in zip(self._ema_model.parameters(), self._model.parameters()):
                # TODO: Perform the exponential moving average, modifying the `ema_variable` in place
                # by multiplying it by `self._ema_momentum` and adding the `variable` multiplied by
                # `(1 - self._ema_momentum)`.
                ...
            return {"loss": self.loss_tracker(loss)}

    @torch.no_grad()
    def generate(self, initial_noise: torch.Tensor, steps: int) -> torch.Tensor:
        images = initial_noise.to(self.device)
        trajectory = []

        # TODO: Perform the sampling process using the Euler method (the one described
        # on the slides) and `steps` number of steps. You should compute:
        # - `images`, which are the final generated images, and
        # - `trajectory`, which is a list of the intermediate images x_0, x_{1/T}, ...,
        #   i.e., all the inputs you passed to the model during this method.

        # Apply the denormalization to the generated images and the trajectory.
        return self.denormalize_image(images), list(map(self.denormalize_image, trajectory))


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        return image, image  # return the image both as the input and the target


class FixedNumberOfSamples(torch.utils.data.Sampler):
    def __init__(self, size: int, samples: int, seed: int) -> None:
        self._size, self._samples, self._permutation = size, samples, []
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self._samples

    def __iter__(self):
        for _ in range(self._samples):
            self._permutation = self._permutation or torch.randperm(self._size, generator=self._generator).tolist()
            yield self._permutation[0]
            self._permutation = self._permutation[1:]


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the image data.
    images64 = Image64Dataset(args.dataset)
    train = TrainableDataset(images64.train).dataloader(args.batch_size, sampler=FixedNumberOfSamples(
        len(images64.train), args.batch_size * args.epoch_batches, args.seed))

    # Create the model.
    flow_matching = FlowMatching(args)

    # Class for sampling images and storing them in TensorBoard.
    class TBSampler:
        def __init__(self, columns: int, rows: int, seed: int) -> None:
            self._columns = columns
            self._rows = rows
            self._noise = torch.randn(
                rows, columns, Image64Dataset.C, Image64Dataset.H, Image64Dataset.W,
                generator=torch.Generator().manual_seed(seed)
            )

        @torch.no_grad()
        def __call__(self, model, epoch, logs) -> None:
            # After the last epoch and every `args.plot_each` epoch, generate a sample to TensorBoard logs.
            if epoch == args.epochs or epoch % (args.plot_each or args.epochs) == 0:
                # Generate a grid of `self._columns *  self._rows` independent samples.
                rows = [model.generate(noise, args.sampling_steps)[0] for noise in list(self._noise)]
                images = torch.cat([torch.cat(list(row), dim=-1) for row in rows], dim=-2)
                model.get_tb_writer("train").add_image("images", images, epoch)
                # Generate gradual denoising process for `rows` samples, showing `self._columns` steps.
                steps = max(1, args.sampling_steps // (self._columns - 1))
                samples, process = model.generate(self._noise[:, 0], steps * (self._columns - 1))
                process = torch.cat([torch.cat(list(col), dim=-2) for col in process[::steps] + [samples]], dim=-1)
                model.get_tb_writer("train").add_image("process", process, epoch)
            # After the last epoch, store statistics of the generated sample for ReCodEx to evaluate.
            if epoch == args.epochs:
                images = images.numpy(force=True)
                logs["sample_mean"], logs["sample_std"] = images.mean(), images.std()

    # Train the model.
    flow_matching.configure(
        optimizer=torch.optim.AdamW(flow_matching.parameters()),
        loss=getattr(torch.nn, args.loss)(),
        logdir=args.logdir,
    )
    logs = flow_matching.fit(train, epochs=args.epochs, callbacks=[TBSampler(16, 10, args.seed)])

    # Return the loss and sample statistics for ReCodEx to validate.
    return logs


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
