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

1. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16`
```
Epoch 1/1 1.8s train_loss=2.2773 train_accuracy=0.3328 dev_loss=1.8841 dev_accuracy=0.4033
```

2. `python3 tagger_cle.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16 --cle_dim=16 --word_masking=0.1`
```
Epoch 1/1 1.8s train_loss=2.2828 train_accuracy=0.3280 dev_loss=1.8993 dev_accuracy=0.4017
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

- `python3 tagger_cle.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=32 --cle_dim=32 --word_masking=0.1`
```
Epoch 1/5 11.3s train_loss=1.1309 train_accuracy=0.6657 dev_loss=0.3800 dev_accuracy=0.9059
Epoch 2/5 10.9s train_loss=0.2212 train_accuracy=0.9422 dev_loss=0.1958 dev_accuracy=0.9453
Epoch 3/5 11.3s train_loss=0.1132 train_accuracy=0.9685 dev_loss=0.2114 dev_accuracy=0.9370
Epoch 4/5 10.8s train_loss=0.0827 train_accuracy=0.9760 dev_loss=0.1642 dev_accuracy=0.9483
Epoch 5/5 11.5s train_loss=0.0651 train_accuracy=0.9801 dev_loss=0.1529 dev_accuracy=0.9532
```
#### Examples End:
