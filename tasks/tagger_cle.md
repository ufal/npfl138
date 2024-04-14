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
Epoch=1/1 4.0s loss=2.2871 accuracy=0.2909 dev_loss=1.8784 dev_accuracy=0.4275
```

2. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch=1/1 4.4s loss=2.2846 accuracy=0.2875 dev_loss=1.8835 dev_accuracy=0.4289
```
#### Tests End:
#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32`
```
Epoch=1/5 22.6s loss=1.0757 accuracy=0.6784 dev_loss=0.3678 dev_accuracy=0.8969
Epoch=2/5 21.5s loss=0.1476 accuracy=0.9684 dev_loss=0.1978 dev_accuracy=0.9375
Epoch=3/5 22.1s loss=0.0490 accuracy=0.9881 dev_loss=0.1722 dev_accuracy=0.9488
Epoch=4/5 21.3s loss=0.0303 accuracy=0.9912 dev_loss=0.1651 dev_accuracy=0.9470
Epoch=5/5 21.1s loss=0.0201 accuracy=0.9942 dev_loss=0.1630 dev_accuracy=0.9479
```

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32 --word_masking=0.1`
```
Epoch=1/5 22.2s loss=1.1264 accuracy=0.6594 dev_loss=0.3980 dev_accuracy=0.8977
Epoch=2/5 21.4s loss=0.2340 accuracy=0.9408 dev_loss=0.2175 dev_accuracy=0.9377
Epoch=3/5 24.1s loss=0.1163 accuracy=0.9690 dev_loss=0.1624 dev_accuracy=0.9525
Epoch=4/5 26.6s loss=0.0852 accuracy=0.9745 dev_loss=0.1493 dev_accuracy=0.9560
Epoch=5/5 24.9s loss=0.0718 accuracy=0.9778 dev_loss=0.1450 dev_accuracy=0.9563
```
#### Examples End:
