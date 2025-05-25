# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""The `TTSDataset` is a collection of (text, mel_spectrogram) pairs.

- The data consists of just one dataset:
    - `train`
- The `train` dataset is a [torch.utils.data.Dataset][] instance providing
    - `__len__`: number of utterances in the dataset;
    - `__getitem__`: return the requested utterance as a dictionary
      with keys:
        - `"text"`: the text of an utterance as a string,
        - `"mel_spectrogram"`: the mel spectrogram of an utterance with shape `[length, n_mels]`;
    - `char_vocab`: a [npfl138.Vocabulary][] instance with the character mapping.
"""
import os
import sys
from typing import TypedDict
import urllib.request
import zipfile

import torch
import torchaudio

from ..vocabulary import Vocabulary


class TTSDataset:
    PAD: int = 0
    """The index of the padding token in the vocabulary."""

    Element = TypedDict("Element", {"text": str, "mel_spectrogram": torch.Tensor})
    """The type of a single dataset element, i.e., a single utterance."""

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, path: str, sample_rate: int, window_length: int, hop_length: int, mels: int) -> None:
            # Load the data
            self._utterances = []
            with open(path, "r", encoding="utf-8") as tsv_file:
                for line in tsv_file:
                    path, text = line.rstrip("\r\n").split("\t")
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
            audio, sample_rate = torchaudio.load(path, normalize=True)  # load audio file
            audio = audio.mean(dim=0)  # convert to mono
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
        """Load the dataset from the given filename, downloading it if necessary."""
        path = "{}.tsv".format(name)
        if not os.path.exists(path):
            zip_path = "{}.zip".format(name)
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, zip_path), filename=zip_path)
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall()
            os.remove(zip_path)

        self.train = self.Dataset(path, sample_rate, window_length, hop_length, mels)

    train: Dataset
    """The training dataset."""
