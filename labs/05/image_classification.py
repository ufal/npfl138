#!/usr/bin/env python3
import argparse
import os
import time
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", type=str, help="Files to classify.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load EfficientNetV2-B0
    efficientnetv2_b0 = keras.applications.EfficientNetV2B0(include_top=True)

    for image_path in args.images:
        # Load the file
        image = keras.utils.load_img(image_path)
        image = keras.utils.img_to_array(image, dtype="uint8")

        # Resize to 224,224
        image = keras.ops.image.resize(image, (224, 224))
        image = keras.applications.efficientnet_v2.preprocess_input(image)

        # Compute the prediction
        start = time.time()

        predictions = efficientnetv2_b0.predict_on_batch(keras.ops.expand_dims(image, 0))

        predictions = keras.applications.efficientnet_v2.decode_predictions(predictions)

        print("Image {} [{} ms] labels:{}".format(
            image_path,
            1000 * (time.time() - start),
            "".join("\n- {}: {}".format(label, prob) for _, label, prob in predictions[0])
        ))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
