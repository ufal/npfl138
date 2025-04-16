### Assignment: lemmatizer_attn
#### Date: Deadline: Apr 30, 22:00
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
  generate weights for every encoder output. Finally sum the encoder outputs
  using these weights and concatenate the computed attention to the decoder
  inputs.

Once submitted to ReCodEx, you should experiment with the effect of using
the attention, and the influence of RNN dimensionality on network performance.

#### Tests Start: lemmatizer_attn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=64 --rnn_dim=32`
```
Epoch 1/1 3.4s train_loss=2.9698 train_accuracy=0.0481 dev_accuracy=0.1655
```

2. `python3 lemmatizer_attn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
Epoch 1/1 3.2s train_loss=2.8633 train_accuracy=0.0313 dev_accuracy=0.1530
```
#### Tests End:
#### Examples Start: lemmatizer_attn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000`
```
Epoch 1/3 27.1s train_loss=1.8938 train_accuracy=0.2621 dev_accuracy=0.5941
Epoch 2/3 36.8s train_loss=0.4003 train_accuracy=0.6906 dev_accuracy=0.7295
Epoch 3/3 29.5s train_loss=0.2510 train_accuracy=0.7734 dev_accuracy=0.7821
```

- `python3 lemmatizer_attn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch 1/3 21.7s train_loss=1.5409 train_accuracy=0.3568 dev_accuracy=0.6244
Epoch 2/3 31.2s train_loss=0.3149 train_accuracy=0.7330 dev_accuracy=0.7685
Epoch 3/3 25.3s train_loss=0.1996 train_accuracy=0.8066 dev_accuracy=0.7966
```
#### Examples End:
