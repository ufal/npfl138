### Assignment: speech_recognition
#### Date: Deadline: Apr 30, 22:00
#### Points: 5 points+5 bonus

This assignment is a competition task in speech recognition area. Specifically,
your goal is to predict a sequence of letters given a spoken utterance.
We will be using Czech recordings from the [Common Voice](https://commonvoice.mozilla.org/),
with input sound waves passed through the usual preprocessing – computing
[Mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
You can repeat this preprocessing on a given audio using the `load_audio` and
`mfcc_extract` methods from the
[common_voice_cs.py](https://github.com/ufal/npfl138/tree/past-2324/labs/09/common_voice_cs.py) module.
This module can also load the dataset, downloading it when necessary (note that
it has 200MB, so it might take a while). Furthermore, you can listen to the
[development portion of the dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/common_voice_cs_dev.html).
Lastly, the whole dataset is available for
[download in MP3 format](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/common_voice_cs_mp3.tar)
(but you are not expected to download that, only if you would like to perform some
custom preprocessing).

Additional following data can be utilized in this assignment:
- You can use any _unannotated_ text data (Wikipedia, Czech National Corpus, …),
  and also any pre-trained word embeddings or language models (assuming they
  were trained on plain texts).
- You can use any _unannotated_ speech data.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions).
The evaluation is performed by computing the edit distance to the gold letter
sequence, normalized by its length (a corresponding metric
`EditDistanceMetric` is provided by the [common_voice_cs.py](https://github.com/ufal/npfl138/tree/past-2324/labs/09/common_voice_cs.py)).
Everyone who submits a solution with at most 50% test set edit distance
gets 5 points; the remaining 5 bonus points are distributed
depending on relative ordering of your solutions. Note that
you can evaluate the predictions as usual using the [common_voice_cs.py](https://github.com/ufal/npfl138/tree/past-2324/labs/08/common_voice_cs.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

Start with the [speech_recognition.py](https://github.com/ufal/npfl138/tree/past-2324/labs/09/speech_recognition.py)
template containing a structure suitable for computing the CTC loss and perform
CTC decoding. You can use [torch.nn.CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
to compute the loss and you can use
[torchaudio.models.decoder.CTCDecoder](https://pytorch.org/audio/main/generated/torchaudio.models.decoder.CTCDecoder.html)/[torchaudio.models.decoder.CUCTCDecoder](https://pytorch.org/audio/main/generated/torchaudio.models.decoder.CUCTCDecoder.html)
to perform beam-search decoding.
