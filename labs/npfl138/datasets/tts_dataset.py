# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The `TTSDataset` is a collection of (text, mel_spectrogram) pairs.

Each dataset element is a Python dictionary with the following keys:

- `"text"`: the text of an utterance as a string,
- `"mel_spectrogram"`: the mel spectrogram of an utterance with shape `[length, n_mels]`;

The data consists of just one split:

- `train`

The `train` dataset is a [torch.utils.data.Dataset][] instance providing:

- `__len__`: number of utterances in the dataset;
- `__getitem__`: return the requested utterance as a dictionary with keys `"text"` and `"mel_spectrogram"`;
- `char_vocab`: a [npfl138.Vocabulary][] instance with the character mapping.
"""
import os
import sys
from typing import TypedDict
import urllib.request
import zipfile

import soundfile
import torch
import torchaudio

from .downloader import download_url_to_file
from ..vocabulary import Vocabulary


class TTSDataset:
    PAD: int = 0
    """The index of the padding token in the vocabulary."""

    Element = TypedDict("Element", {"text": str, "mel_spectrogram": torch.Tensor})
    """The type of a single dataset element, i.e., a single utterance."""

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, sample_rate: int, window_length: int, hop_length: int, mels: int) -> None:
            # Load the data
            self._zip_file = zipfile.ZipFile(path)

            self._utterances = []
            with self._zip_file.open(f"{os.path.basename(path).removesuffix('.zip')}.tsv") as tsv_file:
                for line in tsv_file:
                    path, text = line.decode("utf-8").rstrip("\r\n").split("\t")
                    self._utterances.append((text, path))

            # Create the character mapping
            self._char_vocab = Vocabulary(sorted(set("".join(text for text, _ in self._utterances))), add_unk=True)

            # Create the MelSpectrogram transform
            self._transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=window_length,
                win_length=window_length,
                hop_length=hop_length,
                n_mels=mels,
                power=1,  # Vococ-style mel spectrogram; others like BigVGAN use power=2
            )

        def __len__(self) -> int:
            """Return the number of utterances in the dataset."""
            return len(self._utterances)

        def __getitem__(self, index: int) -> "TTSDataset.Element":
            """Return the `index`-th element of the dataset as a dictionary."""
            text, path = self._utterances[index]
            with self._zip_file.open(path) as audio_file:
                audio, sample_rate = soundfile.read(audio_file, dtype="float32", always_2d=True)  # load audio file
            audio = torch.from_numpy(audio).mean(dim=-1)  # move to PyTorch and convert to mono
            if sample_rate != self._transform.sample_rate:  # resample if necessary
                audio = torchaudio.functional.resample(audio, sample_rate, self._transform.sample_rate)
            mel_spectrogram = self._transform(audio).permute(1, 0)  # mel spectrogram
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-7))  # dynamic range compression
            return {
                "text": text,
                "mel_spectrogram": mel_spectrogram,
            }

        @property
        def char_vocab(self) -> Vocabulary:
            """The character vocabulary of the dataset."""
            return self._char_vocab

    def __init__(self, name: str, sample_rate: int, window_length: int, hop_length: int, mels: int) -> None:
        """Load the dataset from the given filename, downloading it if necessary.

        Parameters:
          name: The name of the dataset, for example `ljspeech_tiny`.
          sample_rate: The sample rate to use for loading the audio files.
          window_length: The window length to use for computing the mel spectrograms.
          hop_length: The hop length to use for computing the mel spectrograms.
          mels: The number of mel bins to use for computing the mel spectrograms.
        """
        path = download_url_to_file(self.URL, f"{name}.zip")
        self.train = self.Dataset(path, sample_rate, window_length, hop_length, mels)

    train: Dataset
    """The training dataset."""
