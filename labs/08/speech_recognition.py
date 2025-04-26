#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchaudio.models.decoder
import torchmetrics

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        ...

    def forward(self, ...) -> torch.Tensor:
        # TODO: Compute the output of the model.
        raise NotImplementedError()

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, ...) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        raise NotImplementedError()

    def ctc_decoding(self, y_pred: torch.Tensor, ...) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        raise NotImplementedError()

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, ...
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        predictions = ...
        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # Note that while the `CommonVoiceCs.LETTER_NAMES` do not explicitly contain a blank token,
        # the [PAD] token can be employed as a blank token.
        raise NotImplementedError()

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        raise NotImplementedError()


def main(args: argparse.Namespace) -> None:
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
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    # TODO: Create the model and train it. The `Model.compute_metrics` method assumes you
    # passed the following metric to the `configure` method under the name "edit_distance":
    #   CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)
    model = ...

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = ...

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
