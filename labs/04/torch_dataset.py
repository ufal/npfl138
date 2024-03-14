#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import numpy as np
import keras
import torch
from torchvision.transforms import v2

from cifar10 import CIFAR10

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_images", default=None, const=10, type=int, nargs="?", help="Show augmented images.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load the data
    cifar = CIFAR10()

    # Create the model
    inputs = keras.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
    hidden = keras.layers.Rescaling(1 / 255)(inputs)
    hidden = keras.layers.Conv2D(16, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(16, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(24, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(24, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(32, 3, 2, "same", activation="relu")(hidden)
    hidden = keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(hidden)
    hidden = keras.layers.Flatten()(hidden)
    hidden = keras.layers.Dense(200, activation="relu")(hidden)
    outputs = keras.layers.Dense(len(CIFAR10.LABELS), activation="softmax")(hidden)

    # Compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # TODO: Create a Torch dataset constructible from the given `CIFAR10.Dataset`.
    # You should use only the first `size` examples of the dataset, and optional
    # augmentation function `augmentation_fn` may be applied to the images.
    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, cifar: CIFAR10.Dataset, size: int, augmentation_fn=None) -> None:
            # TODO: Note that the images and labels are available in `cifar.data["images"]`
            # and `cifar.data["labels"]`.
            ...

        def __len__(self) -> int:
            # TODO: Return the appropriate size.
            ...

        def __getitem__(self, index: int) -> tuple[np.ndarray | torch.Tensor, int]:
            # TODO: Return the `index`-th example from the dataset, with the image optionally
            # passed through the `augmentation_fn` if it is not `None`.
            ...

    if args.augment:
        # Construct a sequence of augmentation transformations from `torchvision.transforms.v2`.
        transformation = v2.Compose([
            # TODO: Add the following transformations:
            # - first create a `v2.RandomResize` that scales the image to
            #   random size in range [28, 36],
            # - then add `v2.Pad` that pads the image with 4 pixels on each side,
            # - then add `v2.RandomCrop` that chooses a random crop of size 32x32,
            # - and finally add `v2.RandomHorizontalFlip` that uniformly
            #   randomly flips the image horizontally.
            ...
        ])

        def augmentation_fn(image: np.ndarray) -> torch.Tensor:
            # TODO: First, convert the numpy `images` to a PyTorch tensor of uint8s,
            # preferably by using `torch.from_numpy` or `torch.as_tensor` to avoid copying.
            # Then, because of the channels-position mismatch, permute the axes
            # in the image to change the order of the axes from HWC to CHW.
            # Next, apply the `transformation` to the image (by calling it with
            # the image as an argument), and finally permute the axes back to
            # the original order.
            return ...
    else:
        augmentation_fn = None

    # TODO: Create `train` and `dev` instances of `TorchDataset` from the corresponding
    # `cifar` datasets. Limit their sizes to 5_000 and 1_000 examples, respectively,
    # and use the `augmentation_fn` for the training dataset.
    train = ...
    dev = ...

    if args.show_images:
        from torch.utils import tensorboard
        GRID, REPEATS, TAG = args.show_images, 5, "augmented" if args.augment else "original"
        tb_writer = tensorboard.SummaryWriter(os.path.join("logs", "augmentations"))
        for step in range(REPEATS):
            images = keras.ops.stack([train[i][0] for i in range(GRID * GRID)], axis=0)
            images = images.reshape(GRID, GRID * images.shape[1], *images.shape[2:]).permute(0, 2, 1, 3)
            images = images.reshape(1, GRID * images.shape[1], *images.shape[2:]).permute(0, 2, 1, 3)
            tb_writer.add_images(TAG, images, step, dataformats="NHWC")
        tb_writer.close()
        print("Saved first {} training imaged to logs/{}".format(GRID * GRID, TAG))

    # TODO: Create `train` and `dev` instances of `torch.utils.data.DataLoader` from
    # the datasets, using the given `args.batch_size` and shuffling the training dataset.
    train = ...
    dev = ...

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev)

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
