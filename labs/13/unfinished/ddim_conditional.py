#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch

from image64_dataset import Image64Dataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--downscale", default=8, type=int, help="Conditional downscale factor.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="MeanAbsoluteError", type=str, help="Loss object to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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


# The diffusion model architecture building blocks.
class SinusoidalEmbedding(keras.layers.Layer):
    """Sinusoidal embeddings used to embed the current noise rate."""
    def __init__(self, dim, *args, **kwargs):
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        super().__init__(*args, **kwargs)
        self.dim = dim

    def call(self, inputs):
        # TODO(ddim): Compute the sinusoidal embeddings of the inputs in `[0, 1]` range.
        # The `inputs` have shape `[..., 1]`, and the embeddings should have
        # a shape `[..., self.dim]`, where for `0 <= i < self.dim/2`,
        # - the value on index `[..., i]` should be
        #     `sin(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        # - the value on index `[..., self.dim/2 + i]` should be
        #     `cos(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        raise NotImplementedError()


def ResidualBlock(inputs, width, noise_embeddings):
    """A residual block with two BN+Swish+3x3Conv, adding noise embeddings in the middle."""
    # TODO(ddim): Compute the residual connection. If the number of filters
    # in the input is the same as `width`, use unmodified `inputs`; otherwise,
    # pass it through a 1x1 convolution with `width` filters.
    residual = ...

    # TODO(ddim): Pass `inputs` through a BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding. As in the `gan` assignment, for simplicity ignore `use_bias=False`.
    hidden = ...

    # TODO(ddim): Pass `noise_embeddings` through a dense layer with `width` outputs and Swish
    # activation, and add it to `hidden`.
    hidden += ...

    # TODO(ddim): Pass `hidden` through another BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding. Furthermore, initialize the kernel of the convolution to all
    # zeros, so that after initialization, the whole residual block is an identity.
    hidden = ...

    hidden += residual
    return hidden


# The DDIM model
class DDIM(keras.Model):
    def __init__(self, args: argparse.Namespace, data: torch.utils.data.Dataset) -> None:
        super().__init__()
        self.built = True

        # Create the network inputs.
        images = keras.Input([Image64Dataset.H, Image64Dataset.W, Image64Dataset.C])
        conditioning = keras.Input(
            [Image64Dataset.H // args.downscale, Image64Dataset.W // args.downscale, Image64Dataset.C])
        noise_rates = keras.Input([1, 1, 1])

        # TODO(ddim): Embed noise rates using the `SinusoidalEmbedding` with `args.channels` dimensions.
        noise_embedding = ...

        # TODO: Upscale the `conditioning` using the `keras.layers.UpSampling2D` by
        # a factor of `args.downscale` with "bicubic" interpolation. Then concatenate
        # the input images and the upscaled conditioning, and pass the result through
        # an initial 3x3 convolution with `args.channels` filters and "same" padding.
        hidden = ...

        # Downscaling stages
        outputs = []
        for i in range(args.stages):
            # TODO(ddim): For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
            # with `args.channels << i` filters and with the `noise_embedding`, and append
            # every result to the `outputs` array.
            ...

            # TODO(ddim): Downscale `hidden` with a 3x3 convolution with stride 2,
            # `args.channels << (i + 1)` filters, and "same" padding.
            hidden = ...

        # Middle block
        # TODO(ddim): For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
        # with `args.channels << args.stages` filters.
        ...

        # Upscaling stages
        for i in reversed(range(args.stages)):
            # TODO(ddim): Upscale `hidden` with a 4x4 transposed convolution with stride 2,
            # `args.channels << i` filters, and "same" padding.
            hidden = ...

            # TODO(ddim): For `args.stage_blocks` times, concatenate `hidden` and `outputs.pop()`,
            # and pass the result through a `ResidualBlock` with `args.channels << i` filters.
            ...

        # Verify that all outputs have been used.
        assert len(outputs) == 0

        # TODO(ddim): Compute the final output by passing `hidden` through a
        # BatchNormalization, Swish activation, and a 3x3 convolution with
        # `Image64Dataset.C` channels and "same" padding, with kernel of
        # the convolution initialized to all zeros.
        outputs = ...

        self._network = keras.Model(inputs=[images, conditioning, noise_rates], outputs=outputs)

        # Create the EMA network, which will be updated by exponential moving averaging.
        self._ema_network = keras.models.clone_model(self._network)

        # Compute image normalization statistics.
        first_moment, second_moment, count = 0, 0, 0
        for image in data:
            image = image.to(torch.float32)
            first_moment += torch.sum(image)
            second_moment += torch.sum(image * image)
            count += image.numel()
        self._image_normalization_mean = first_moment / count
        self._image_normalization_sd = torch.sqrt(second_moment / count - self._image_normalization_mean ** 2)

        # Store required arguments for later usage.
        self._downscale = args.downscale
        self._ema_momentum = args.ema
        self._seed = args.seed

        self.tb_callback = TorchTensorBoardCallback(args.logdir)

    def _image_normalization(self, images):
        """Normalize the images to have zero mean and unit variance."""
        images = (images.to(torch.float32) - self._image_normalization_mean) / self._image_normalization_sd
        return images

    def _image_denormalization(self, images):
        """Invert the `self._image_normalization`, returning an image represented using bytes."""
        images = self._image_normalization_mean + images * self._image_normalization_sd
        images = torch.clamp(torch.round(images), 0, 255).to(torch.uint8)
        return images

    def _diffusion_rates(self, times):
        """Compute signal and noise rates for the given times."""
        starting_angle, final_angle = 0.025, np.pi / 2 - 0.025
        # TODO(ddim): For a vector of `times` in [0, 1] range, return a pair of corresponding
        # `(signal_rates, noise_rates)`. The signal and noise rates are computed as
        # cosine and sine of an angle which is a linear interpolation from `starting_angle`
        # of 0.025 rad (for time 0) to `final_angle` of pi/2 - 0.025 rad (for time 1).
        # Because we use the rates as multipliers of image batches, reshape the rates
        # to a shape `[batch_size, 1, 1, 1]`, assuming `times` has a shape `[batch_size]`.
        signal_rates, noise_rates = ...

        return signal_rates, noise_rates

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Perform a training step."""
        # Normalize the images so have on average zero mean and unit variance.
        images = self._image_normalization(images)
        # TODO: Compute the conditioning by using the `keras.layers.AveragePooling2D`
        # layer downscaling the `images` by a factor of `self._downscale`.
        conditioning = ...

        # Generate a random noise of the same shape as the `images`.
        noises = torch.randn(images.shape)
        # Generate a batch of times when to perform the loss computation in.
        times = torch.rand(images.shape[:1])

        # TODO(ddim): Compute the signal and noise rates using the sampled `times`.
        signal_rates, noise_rates = ...

        # TODO(ddim): Compute the noisy images utilizing the computed signal and noise rates.
        noisy_images = ...

        # TODO: Predict the noise by running the `self._network` on the noisy images,
        # the conditioning, and the noise rates. Do not forget to also pass the
        # `training=True` argument (to run batch normalizations in training regime).
        predicted_noises = ...

        # TODO(ddim): Compute loss using the `self.compute_loss`.
        loss = ...

        # Perform an update step.
        self.zero_grad()
        loss.backward()
        self.optimizer.apply([v.value.grad for v in self.trainable_variables], self.trainable_variables)

        # Update the `self._ema_network` using exponential moving average.
        for ema_variable, variable in zip(self._ema_network.variables, self._network.variables):
            ema_variable.assign(self._ema_momentum * ema_variable + (1 - self._ema_momentum) * variable)

        return {metric.name: metric.result() for metric in self.metrics}

    def generate(self, initial_noise, conditioning, steps):
        """Sample a batch of images given the `initial_noise` using `steps` steps."""
        images = initial_noise
        diffusion_process = []

        # TODO: Normalize the `conditioning` using the `self._image_normalization`.
        conditioning = ...

        # We emply a uniformly distributed sequence of times from 1 to 0. We in fact
        # create an identical batch of them, and we also make the time of the next step
        # available in the body of the cycle, because it is needed by the DDIM algorithm.
        steps = torch.linspace(1., 0., steps + 1).unsqueeze(-1).repeat(1, images.shape[0])

        for times, next_times in zip(steps[:-1], steps[1:]):
            # Store the current images converted to `torch.uint8` to allow denoising visualization.
            diffusion_process.append(self._image_denormalization(images))

            # TODO(ddim): Compute the signal and noise rates of the current time step.
            signal_rates, noise_rates = ...

            # TODO: Predict the noise by calling the `self._ema_network` with `training=False`.
            predicted_noises = ...

            # TODO(ddim): Predict the denoised version of `images` (i.e., the $x_0$ estimate
            # in the DDIM sampling algorithm).
            denoised_images = ...

            # TODO(ddim): Compute the signal and noise rates of the next time step.
            next_signal_rates, next_noise_rates = ...

            # TODO(ddim): Update the `images` according to the DDIM sampling algorithm.
            images = ...

        # TODO(ddim): Compute the output by passing the latest `denoised_images` through
        # the `self._image_denormalization` to obtain a `torch.uint8` representation.
        images = ...

        return images, diffusion_process


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

    # Load the image data. Keep the first 80 images
    # as conditioning used for generation in `TBSampler`.
    images64 = Image64Dataset(args.dataset)
    train = images64.train.transform(lambda example: example["image"])
    dev = torch.stack([train[index] for index in range(80)])
    # TODO: Using `dev` (a batch of 80 images), compute the conditioning by downscaling
    # it using `keras.layers.AveragePooling2D` by a factor of `args.downscale`.
    # Because the images are represented using `torch.uint8`, you need to convert them to
    # `torch.float32` first (by casting) and then back to bytes (by `torch.round` and a cast).
    conditioning = ...

    # Create the model; the image data are used to initialize the image normalization layer.
    ddim = DDIM(args, train)

    # Create the data pipeline.
    class FixedNumberOfBatches(torch.utils.data.Sampler):
        def __init__(self, offset: int, dataset_length: int, samples: int, seed: int) -> None:
            self._offset = offset
            self._dataset_length = dataset_length
            self._samples = samples
            self._generator = torch.Generator().manual_seed(seed)
            self._permutation = []

        def __len__(self):
            return self._samples

        def __iter__(self):
            for _ in range(self._samples):
                if not len(self._permutation):
                    self._permutation = torch.randperm(self._dataset_length - self._offset, generator=self._generator)
                yield self._offset + self._permutation[0]
                self._permutation = self._permutation[1:]

    train = torch.utils.data.DataLoader(train, batch_size=args.batch_size, sampler=FixedNumberOfBatches(
        80, len(train), args.epoch_batches * args.batch_size, args.seed))

    # Class for sampling images and storing them to TensorBoard.
    class TBSampler:
        def __init__(self, columns: int, rows: int) -> None:
            self._columns = columns
            self._rows = rows
            self._noise = torch.randn([columns * rows, Image64Dataset.H, Image64Dataset.W, Image64Dataset.C])

        def __call__(self, epoch, logs=None) -> None:
            # After the last epoch and every `args.plot_each` epoch, generate a sample to TensorBoard logs.
            if epoch + 1 == args.epochs or (epoch + 1) % (args.plot_each or args.epochs) == 0:
                # Generate a grid of `self._columns *  self._rows // 2` independent samples with conditioning.
                samples = self._columns * self._rows // 2
                images, _ = ddim.generate(self._noise[:samples], conditioning[:samples], args.sampling_steps)
                image_rows = [torch.cat(list(row), axis=1) for row in torch.chunk(images, self._rows // 2)]
                conditioning_rows = [torch.cat(list(row), axis=1) for row in torch.chunk(
                    keras.layers.UpSampling2D(args.downscale)(conditioning[:samples]), self._rows // 2)]
                images = torch.cat([row for rows in zip(conditioning_rows, image_rows) for row in rows], axis=0)
                ddim.tb_callback.writer("train").add_image("images", images, epoch + 1, dataformats="HWC")
                # For each of `self._columns *  self._rows // 5` conditionings, generate 3 different samples.
                samples = self._columns * self._rows // 5
                images, _ = ddim.generate(self._noise[:3 * samples],
                                          torch.repeat(conditioning[:samples], 3, axis=0), args.sampling_steps)
                image_rows = [torch.cat([torch.cat(list(col), axis=0) for col in torch.chunk(row, self._columns)], axis=1)
                              for row in torch.chunk(images, self._rows // 5)]
                dev_rows = [torch.cat(list(row), axis=1) for row in torch.chunk(dev[:samples], self._rows // 5)]
                conditioning_rows = [torch.cat(list(row), axis=1) for row in torch.chunk(
                    keras.layers.UpSampling2D(args.downscale)(conditioning[:samples]), self._rows // 5)]
                variants = torch.cat([row for rows in zip(dev_rows, conditioning_rows, image_rows) for row in rows], axis=0)
                ddim.tb_callback.writer("train").add_image("variants", variants, epoch + 1, dataformats="HWC")
            # After the last epoch, store statistics of the generated sample for ReCodEx to evaluate.
            if epoch + 1 == args.epochs:
                logs["sample_mean"] = torch.mean(images.to(torch.float32))
                logs["sample_std"] = torch.std(images.to(torch.float32))

    # Train the model
    ddim.compile(
        optimizer=keras.optimizers.AdamW(),
        loss=getattr(keras.losses, args.loss)(),
    )
    logs = ddim.fit(train, epochs=args.epochs, callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=TBSampler(16, 10)), ddim.tb_callback])

    # Return the loss and sample statistics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
