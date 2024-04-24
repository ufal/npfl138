### Assignment: lemmatizer_attn
#### Date: Deadline: May 7, 22:00
#### Points: 3 points
#### Tests: lemmatizer_attn_tests
#### Examples: lemmatizer_attn_examples

This task is a continuation of the `lemmatizer_noattn` assignment. Using the
[lemmatizer_attn.py](https://github.com/ufal/npfl138/tree/master/labs/10/lemmatizer_attn.py)
template, implement the following features in addition to `lemmatizer_noattn`:
- The bidirectional GRU encoder returns outputs for all input characters, not
  just the last.
- Implement attention in the decoders. Notably, project the encoder outputs and
  current state into same-dimensionality vectors, apply non-linearity, and
  generate weights for every encoder output. Finally sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.

#### Tests Start: lemmatizer_attn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
Epoch=1/1 7.3s loss=3.0203 accuracy=0.0016 dev_accuracy=0.0338
```

2. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
Epoch=1/1 6.9s loss=2.8839 accuracy=0.0362 dev_accuracy=0.1570
```
#### Tests End:
#### Examples Start: lemmatizer_attn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000`
```
Epoch=1/3 43.9s loss=1.9892 accuracy=0.2314 dev_accuracy=0.5919
Epoch=2/3 44.3s loss=0.3911 accuracy=0.7048 dev_accuracy=0.7501
Epoch=3/3 46.0s loss=0.2234 accuracy=0.7894 dev_accuracy=0.7836
```

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch=1/3 44.3s loss=1.5970 accuracy=0.3400 dev_accuracy=0.6315
Epoch=2/3 45.0s loss=0.3174 accuracy=0.7320 dev_accuracy=0.7521
Epoch=3/3 45.2s loss=0.1880 accuracy=0.8076 dev_accuracy=0.8040
```
#### Examples End:
