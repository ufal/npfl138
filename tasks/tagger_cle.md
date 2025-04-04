### Assignment: tagger_cle
#### Date: Deadline: Apr 16, 22:00
#### Points: 3 points
#### Tests: tagger_cle_tests
#### Examples: tagger_cle_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_cle.py](https://github.com/ufal/npfl138/tree/master/labs/07/tagger_cle.py)
template, implement character-level word embedding computation using
a bidirectional character-level GRU.

Once submitted to ReCodEx, you should experiment with the effect of CLEs
compared to a plain `tagger_we`, and the influence of their dimensionality. Note
that `tagger_cle` has by default smaller word embeddings so that the size
of word representation (64 + 32 + 32) is the same as in the `tagger_we` assignment.

_Again, in the alternative [tagger_cle.packed.py](https://github.com/ufal/npfl138/tree/master/labs/07/tagger_cle.packed.py)
template, `forward` processes `PackedSequence`s instead of rectangular tensors
and produces also a `PackedSequence`; both templates deliver the same results
when word masking is not used, and are both accepted by ReCodEx._

#### Tests Start: tagger_cle_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=24`
```
Epoch 1/1 2.0s train_loss=2.2294 train_accuracy=0.3722 dev_loss=1.8014 dev_accuracy=0.4973
```

2. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=GRU --rnn_dim=16 --cle_dim=24 --word_masking=0.1`
```
Epoch 1/1 1.9s train_loss=2.0588 train_accuracy=0.4126 dev_loss=1.4207 dev_accuracy=0.5601
```
#### Tests End:
#### Examples Start: tagger_cle_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32`
```
Epoch 1/5 11.2s train_loss=1.0658 train_accuracy=0.6894 dev_loss=0.3472 dev_accuracy=0.9141
Epoch 2/5 10.9s train_loss=0.1342 train_accuracy=0.9714 dev_loss=0.1787 dev_accuracy=0.9469
Epoch 3/5 10.9s train_loss=0.0477 train_accuracy=0.9889 dev_loss=0.1627 dev_accuracy=0.9475
Epoch 4/5 11.0s train_loss=0.0296 train_accuracy=0.9923 dev_loss=0.1712 dev_accuracy=0.9393
Epoch 5/5 10.9s train_loss=0.0197 train_accuracy=0.9952 dev_loss=0.1713 dev_accuracy=0.9474
```

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=GRU --rnn_dim=32 --cle_dim=32 --word_masking=0.1`
```
Epoch 1/5 11.2s train_loss=0.8103 train_accuracy=0.7639 dev_loss=0.2349 dev_accuracy=0.9326
Epoch 2/5 10.8s train_loss=0.1409 train_accuracy=0.9599 dev_loss=0.1599 dev_accuracy=0.9493
Epoch 3/5 11.0s train_loss=0.0807 train_accuracy=0.9750 dev_loss=0.1511 dev_accuracy=0.9529
Epoch 4/5 11.1s train_loss=0.0613 train_accuracy=0.9800 dev_loss=0.1363 dev_accuracy=0.9562
Epoch 5/5 11.1s train_loss=0.0534 train_accuracy=0.9825 dev_loss=0.1499 dev_accuracy=0.9533
```
#### Examples End:
