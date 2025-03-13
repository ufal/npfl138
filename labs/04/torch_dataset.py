#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import torchvision
from torchvision.transforms import v2

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_images", default=None, const=10, type=int, nargs="?", help="Show augmented images.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# Create a dataset class preparing the data and optionally performing the given augmentation.
class ManualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None) -> None:
        self._dataset = dataset
        self._augmentation_fn = augmentation_fn

    def __len__(self) -> int:
        # TODO: Return the length of the dataset; you can use `len` on the `self._dataset`.
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Start by indexing the `self._dataset` with `index` to get
        # the `index`-th example. It is a dictionary with keys "image" and "label".
        # Return an (image, label)` pair, where
        # - the original image needs to be converted to `torch.float32`, divided
        #   by 255, and passed through `self._augmentation_fn` if it is not `None`;
        # - the label is passed unchanged.
        ...


# We can also make our life slightly easier by using the `npfl138.TransformedDataset`.
class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Here the `example` is already the selected example from the underlying
        # dataset; you now only need to process it as in the `ManualDataset`, so
        # (1) convert to `torch.float32`, (2) divide by 255, and (3) apply the
        # `self._augmentation_fn` if it is not `None`; finally, return (image, label) pair.
        ...

    # Furthermore, we could also define a batch-wise transformation function
    #   def transform_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # However, the dataloader then must be created using the `TransformedDataset.dataloader` method,
    # otherwise the `transform_batch` method would not be called.


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

    # Load the data.
    cifar = CIFAR10(sizes={"train": 5_000, "dev": 1_000})

    # Create the model.
    model = npfl138.TrainableModule(torch.nn.Sequential(
        torch.nn.LazyConv2d(16, 3, 2, 1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(16, 3, 1, 1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(24, 3, 2, 1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(24, 3, 1, 1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(32, 3, 2, 1), torch.nn.ReLU(),
        torch.nn.LazyConv2d(32, 3, 1, 1), torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(200), torch.nn.ReLU(),
        torch.nn.LazyLinear(CIFAR10.LABELS),
    ))
    model.eval()(torch.zeros(1, CIFAR10.C, CIFAR10.H, CIFAR10.W))  # Construct the lazy layers with dummy input.

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
        logdir=args.logdir,
    )

    if args.augment:
        # Construct a sequence of augmentation transformations from `torchvision.transforms.v2`.
        augmentation_fn = v2.Compose([
            # TODO: Add the following transformations:
            # - first create a `v2.RandomResize` that scales the image to
            #   random size in range [28, 36],
            # - then add `v2.Pad` that pads the image with 4 pixels on each side,
            # - then add `v2.RandomCrop` that chooses a random crop of size 32x32,
            # - and finally add `v2.RandomHorizontalFlip` that uniformly
            #   randomly flips the image horizontally.
            ...
        ])
    else:
        augmentation_fn = None

    # We now create the dataset; we use both the `ManualDataset` and the `TransformedDataset`,
    # but in practice you would use only one of them.
    train = ManualDataset(cifar.train, augmentation_fn)
    dev = TransformedDataset(cifar.dev)

    if args.show_images:
        GRID, REPEATS, TAG = args.show_images, 5, "augmented" if args.augment else "original"
        for step in range(REPEATS):
            grid = torchvision.utils.make_grid([train[i][0] for i in range(GRID * GRID)], nrow=GRID)
            model.get_tb_writer("train").add_image(TAG, grid, step)
        print("Saved first {} training imaged to logs/{}".format(GRID * GRID, TAG))

    # We now create the `torch.utils.data.DataLoader` instances. For the `train` dataset,
    # we create it manually, for `dev` we use the `TransformedDataset.dataloader`.
    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.dataloader_workers, persistent_workers=args.dataloader_workers > 0)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)

    # Train the model.
    logs = model.fit(train, dev=dev, epochs=args.epochs, log_graph=True)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
