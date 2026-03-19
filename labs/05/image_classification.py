#!/usr/bin/env python3
import argparse
import time

import timm
import torch
import torchvision
import torchvision.transforms.v2 as v2

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", type=str, help="Files to classify.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the number of threads.
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load the EfficientNetV2-B0 model.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True).eval()

    # Create a simple preprocessing pipeline.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Resize(224, interpolation=v2.InterpolationMode(efficientnetv2_b0.pretrained_cfg["interpolation"])),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # Load the ImageNet labels.
    imagenet_labels = timm.data.ImageNetInfo().label_descriptions()

    for image_path in args.images:
        # Load the image.
        image = torchvision.io.decode_image(image_path, mode="RGB")

        # Transform the image by resizing to 224, 224 and normalizing.
        image = preprocessing(image)

        # Compute the prediction
        start = time.time()

        with torch.no_grad():
            predictions = efficientnetv2_b0(image.unsqueeze(0)).squeeze(0)

        predictions = torch.topk(predictions.softmax(dim=-1), k=5)

        print("Image {} [{} ms] labels:{}".format(
            image_path,
            1000 * (time.time() - start),
            "".join("\n- {}: {}".format(imagenet_labels[label], prob) for prob, label in zip(*predictions)),
        ))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
