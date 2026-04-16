### Assignment: lemmatizer_attn
#### Date: Deadline: Apr 29, 22:00
#### Points: 3 points
#### Tests: lemmatizer_attn_tests
#### Examples: lemmatizer_attn_examples

This task is a continuation of the `lemmatizer_noattn` assignment. Using the
[lemmatizer_attn.py](https://github.com/ufal/npfl138/tree/master/labs/09/lemmatizer_attn.py)
template, implement the following features in addition to `lemmatizer_noattn`:
- The bidirectional GRU encoder returns outputs for all input characters, not
  just the last.
- Implement attention in the decoders. Notably, project the encoder outputs and
  current state into same-dimensionality vectors, apply non-linearity, and
  generate weights for every encoder output (masking logits corresponding
  to padding input elements by -1e9). Finally, sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.

#### Tests Start: lemmatizer_attn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 lemmatizer_attn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=64 --rnn_dim=32`
```
Epoch 1/1 5.4s loss=2.9179 accuracy=0.0483 dev:accuracy=0.2045
```

2. `python3 lemmatizer_attn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
Epoch 1/1 5.3s loss=2.8749 accuracy=0.0316 dev:accuracy=0.1838
```
#### Tests End:
#### Examples Start: lemmatizer_attn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000`
```
Epoch 1/3 27.9s loss=1.8950 accuracy=0.2587 dev:accuracy=0.5991
Epoch 2/3 27.4s loss=0.3815 accuracy=0.7033 dev:accuracy=0.7515
Epoch 3/3 27.3s loss=0.2258 accuracy=0.7861 dev:accuracy=0.8032
```

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch 1/3 27.4s loss=1.5573 accuracy=0.3540 dev:accuracy=0.6364
Epoch 2/3 27.6s loss=0.2994 accuracy=0.7487 dev:accuracy=0.7608
Epoch 3/3 27.4s loss=0.1855 accuracy=0.8161 dev:accuracy=0.8132
```
#### Examples End:
