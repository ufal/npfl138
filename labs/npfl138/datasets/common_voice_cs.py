# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The CommonVoiceCs is a Czech subset of the spoken Common Voice dataset.

The task is to transcribe a given audio sample into a sentence. The dataset contains
recordings of people speaking in Czech, with input sound waves passed through
the usual preprocessing—computing
[Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
You can repeat this preprocessing on a given audio using the
[load_audio][npfl138.datasets.common_voice_cs.CommonVoiceCs.load_audio]
and [extract_mfcc][npfl138.datasets.common_voice_cs.CommonVoiceCs.mfcc_extract] methods.

The dataset is automatically downloaded if necessary, but note that is has
200MB, so it might take a while. Furthermore, you can listen to the [development portion of
the dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/demos/common_voice_cs_dev.html).

Each dataset element is a Python dictionary with the following keys:

- `"mfccs"`: a sequence of MFCCs with shape `[sequence_length, CommonVoiceCs.MFCC_DIM=13]`,
- `"sentence"`: a string with the transcription of the audio sample.

The dataset is split into:

- `train`: 9,773 utterances for training;
- `dev`: 904 utterances for development (validation);
- `test`: 3,240 utterances for testing.
"""
from typing import Sequence, TextIO, TypedDict

import torch
import torchaudio

from .. import metrics
from .downloader import download_url_to_file
from .tfrecord_dataset import TFRecordDataset
from ..vocabulary import Vocabulary


class CommonVoiceCs:
    PAD: int = 0
    """The index of the padding token in the vocabulary."""

    MFCC_DIM: int = 13
    """The dimensionality of the MFCC features."""

    LETTERS: int = 48
    """The number of letters used in the dataset."""
    LETTER_NAMES: list[str] = [
        "[PAD]",
        " ", "a", "á", "ä", "b", "c", "č", "d", "ď", "e", "é", "è", "ě", "f", "g", "h",
        "i", "í", "ï", "j", "k", "l", "m", "n", "ň", "o", "ó", "ö", "p", "q", "r", "ř",
        "s", "š", "t", "ť", "u", "ú", "ů", "ü", "v", "w", "x", "y", "ý", "z", "ž",
    ]
    """The list of letter strings used in the dataset."""
    LETTERS_VOCAB: Vocabulary = Vocabulary(LETTER_NAMES)
    """The [npfl138.Vocabulary][] object of the letters used in the dataset."""

    Element = TypedDict("Element", {"mfccs": torch.Tensor, "sentence": str})
    """The type of a single dataset element."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets"

    class Dataset(TFRecordDataset):
        def __init__(self, path: str, size: int, decode_on_demand: bool) -> None:
            super().__init__(path, size, decode_on_demand)

        def __len__(self) -> int:
            """Return the number of elements in the dataset."""
            return super().__len__()

        def __getitem__(self, index: int) -> "CommonVoiceCs.Element":
            """Return the `index`-th element of the dataset."""
            return super().__getitem__(index)

        def _tfrecord_decode(self, data: dict, indices: dict, index: int) -> "CommonVoiceCs.Element":
            return {
                "mfccs": data["mfccs"][indices["mfccs"][index]:indices["mfccs"][index + 1]].view(
                    -1, CommonVoiceCs.MFCC_DIM),
                "sentence": data["sentence"][
                    indices["sentence"][index]:indices["sentence"][index + 1]].numpy().tobytes().decode("utf-8"),
            }

    def __init__(self, decode_on_demand: bool = False) -> None:
        "Load the CommonVoiceCs dataset, downloading it if necessary."
        for dataset, size in [("train", 9_773), ("dev", 904), ("test", 3_240)]:
            path = download_url_to_file(self.URL, f"common_voice_cs.{dataset}.tfrecord")
            setattr(self, dataset, self.Dataset(path, size, decode_on_demand))

    train: Dataset
    """The training dataset."""
    dev: Dataset
    """The development dataset."""
    test: Dataset
    """The test dataset."""

    # Methods for generating MFCC features.
    def load_audio(self, path: str, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
        """Load an audio file and return the audio tensor and sample rate.

        Optionally resample the audio to the target sample rate.
        """
        audio, sample_rate = torchaudio.load(path)
        if target_sample_rate is not None and target_sample_rate != sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        return torch.mean(audio, dim=0), sample_rate

    def mfcc_extract(self, audio: torch.Tensor, sample_rate: int = 16_000) -> torch.Tensor:
        """Extract MFCC features from an audio tensor.

        This function can be used to extract MFCC features from any audio sample,
        allowing to perform speech recognition on any audio sample.
        """
        assert sample_rate == 16_000, "Only 16k sample rate is supported"

        if not hasattr(self, "_mfcc_fn"):
            # Compute a 1024-point STFT with frames of 64 ms and 75% overlap.
            # Then warp the linear scale spectrograms into the mel-scale.
            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            # Finally, compute MFCCs from log-mel-spectrograms and take the first
            # `CommonVoiceCs.MFCC_DIM=13` of them.
            self._mfcc_fn = torchaudio.transforms.MFCC(
                sample_rate=16_000, n_mfcc=self.MFCC_DIM, log_mels=True,
                melkwargs={"n_fft": 1024, "win_length": 1024, "hop_length": 256,
                           "f_min": 80., "f_max": 7600., "n_mels": 80, "center": False}
            )

        # Compute MFCCs of shape `[sequence_length, CommonVoiceCs.MFCC_DIM=13]`.
        mfccs = self._mfcc_fn(audio).permute(1, 0)

        # Scale the first MFCC coefficient by for consistency with existing CommonVoiceCs MFCCs.
        mfccs[:, 0] *= 2**0.5
        return mfccs

    # The EditDistanceMetric
    EditDistanceMetric = metrics.EditDistance
    """The edit distance metric used for evaluation."""

    # Evaluation infrastructure
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[str]) -> float:
        """Evaluate the `predictions` against the gold dataset.

        Returns:
          edit_distance: The average edit distance of the predictions.
        """
        gold = [example["sentence"] for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        return metrics.EditDistance().update(predictions, gold).compute()

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        """Evaluate the file with predictions against the gold dataset.

        Returns:
          edit_distance: The average edit distance of the predictions.
        """
        predictions = []
        for line in predictions_file:
            predictions.append(line.rstrip("\n"))
        return CommonVoiceCs.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            edit_distance = CommonVoiceCs.evaluate_file(getattr(CommonVoiceCs(), args.dataset), predictions_file)
        print(f"CommonVoiceCs edit distance: {100 * edit_distance:.2f}%")
