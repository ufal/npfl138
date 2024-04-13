### Assignment: tagger_cle
#### Date: Deadline: Apr 23, 22:00
#### Points: 3 points
#### Tests: tagger_cle_tests
#### Examples: tagger_cle_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle.py](https://github.com/ufal/npfl138/tree/master/labs/08/tagger_cle.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_cle` has by default smaller word embeddings so that the size
of word representation (64 + 32 + 32) is the same as in the `tagger_we` assignment.

#### Tests Start: tagger_cle_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16`
```
Epoch=1/1 4.0s loss=2.3186 accuracy=0.2965 dev_loss=2.0280 dev_accuracy=0.3561
```

2. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch=1/1 4.0s loss=2.3164 accuracy=0.2849 dev_loss=2.0473 dev_accuracy=0.3582
```
#### Tests End:
#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32`
```
Epoch=1/5 22.3s loss=1.0977 accuracy=0.6714 dev_loss=0.3817 dev_accuracy=0.8926
Epoch=2/5 21.4s loss=0.1437 accuracy=0.9687 dev_loss=0.1959 dev_accuracy=0.9401
Epoch=3/5 21.4s loss=0.0453 accuracy=0.9888 dev_loss=0.1739 dev_accuracy=0.9447
Epoch=4/5 21.5s loss=0.0295 accuracy=0.9915 dev_loss=0.1722 dev_accuracy=0.9463
Epoch=5/5 21.5s loss=0.0210 accuracy=0.9940 dev_loss=0.1741 dev_accuracy=0.9447
```

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32 --word_masking=0.1`
```
Epoch=1/5 22.5s loss=1.1478 accuracy=0.6509 dev_loss=0.4230 dev_accuracy=0.8815
Epoch=2/5 21.8s loss=0.2436 accuracy=0.9356 dev_loss=0.2128 dev_accuracy=0.9356
Epoch=3/5 21.9s loss=0.1193 accuracy=0.9669 dev_loss=0.1819 dev_accuracy=0.9428
Epoch=4/5 21.9s loss=0.0905 accuracy=0.9729 dev_loss=0.1630 dev_accuracy=0.9491
Epoch=5/5 21.8s loss=0.0754 accuracy=0.9765 dev_loss=0.1549 dev_accuracy=0.9548
```
#### Examples End:
