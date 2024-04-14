### Assignment: sequence_classification
#### Date: Deadline: Apr 23, 22:00
#### Points: 2 points
#### Tests: sequence_classification_tests
#### Examples: sequence_classification_examples

The goal of this assignment is to introduce recurrent neural networks, show
their convergence speed, and illustrate exploding gradient issue. The network
should process sequences of 50 small integers and compute parity for each prefix
of the sequence. The inputs are either 0/1, or vectors with one-hot
representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl138/tree/master/labs/08/sequence_classification.py)
template and implement the following:
- Use the specified RNN type (`SimpleRNN`, `GRU`, and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard.
Concentrate on the way how the RNNs converge, convergence speed, exploding
gradient issues and how gradient clipping helps:
- `--rnn=SimpleRNN --sequence_dim=1`, `--rnn=GRU --sequence_dim=1`, `--rnn=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=3`
- the same as above but with `--sequence_dim=10`
- `--rnn=SimpleRNN --hidden_layer=85 --rnn_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn=GRU` with and without `--clip_gradient=1`
- the same as above but with `--rnn=LSTM` with and without `--clip_gradient=1`

#### Tests Start: sequence_classification_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=SimpleRNN --epochs=5`
```
Epoch 1/5 accuracy: 0.4854 - loss: 0.7253 - val_accuracy: 0.5092 - val_loss: 0.6971
Epoch 2/5 accuracy: 0.5101 - loss: 0.6944 - val_accuracy: 0.4990 - val_loss: 0.6914
Epoch 3/5 accuracy: 0.5000 - loss: 0.6904 - val_accuracy: 0.5198 - val_loss: 0.6892
Epoch 4/5 accuracy: 0.5200 - loss: 0.6887 - val_accuracy: 0.5328 - val_loss: 0.6875
Epoch 5/5 accuracy: 0.5326 - loss: 0.6869 - val_accuracy: 0.5362 - val_loss: 0.6857
```

2. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=GRU --epochs=5`
```
Epoch 1/5 accuracy: 0.5277 - loss: 0.6925 - val_accuracy: 0.5217 - val_loss: 0.6921
Epoch 2/5 accuracy: 0.5183 - loss: 0.6921 - val_accuracy: 0.5217 - val_loss: 0.6918
Epoch 3/5 accuracy: 0.5185 - loss: 0.6919 - val_accuracy: 0.5217 - val_loss: 0.6914
Epoch 4/5 accuracy: 0.5212 - loss: 0.6914 - val_accuracy: 0.5282 - val_loss: 0.6910
Epoch 5/5 accuracy: 0.5320 - loss: 0.6904 - val_accuracy: 0.5355 - val_loss: 0.6905
```

3. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5`
```
Epoch 1/5 accuracy: 0.5359 - loss: 0.6926 - val_accuracy: 0.5361 - val_loss: 0.6925
Epoch 2/5 accuracy: 0.5358 - loss: 0.6925 - val_accuracy: 0.5333 - val_loss: 0.6923
Epoch 3/5 accuracy: 0.5370 - loss: 0.6923 - val_accuracy: 0.5369 - val_loss: 0.6920
Epoch 4/5 accuracy: 0.5342 - loss: 0.6919 - val_accuracy: 0.5366 - val_loss: 0.6917
Epoch 5/5 accuracy: 0.5378 - loss: 0.6915 - val_accuracy: 0.5444 - val_loss: 0.6914
```

4. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --hidden_layer=50`
```
Epoch 1/5 accuracy: 0.5377 - loss: 0.6923 - val_accuracy: 0.5414 - val_loss: 0.6911
Epoch 2/5 accuracy: 0.5465 - loss: 0.6902 - val_accuracy: 0.5577 - val_loss: 0.6878
Epoch 3/5 accuracy: 0.5600 - loss: 0.6862 - val_accuracy: 0.5450 - val_loss: 0.6811
Epoch 4/5 accuracy: 0.5491 - loss: 0.6783 - val_accuracy: 0.5590 - val_loss: 0.6707
Epoch 5/5 accuracy: 0.5539 - loss: 0.6678 - val_accuracy: 0.5433 - val_loss: 0.6591
```

5. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --hidden_layer=50 --clip_gradient=0.01`
```
Epoch 1/5 accuracy: 0.5421 - loss: 0.6923 - val_accuracy: 0.5409 - val_loss: 0.6910
Epoch 2/5 accuracy: 0.5504 - loss: 0.6900 - val_accuracy: 0.5511 - val_loss: 0.6875
Epoch 3/5 accuracy: 0.5566 - loss: 0.6860 - val_accuracy: 0.5494 - val_loss: 0.6816
Epoch 4/5 accuracy: 0.5504 - loss: 0.6788 - val_accuracy: 0.5398 - val_loss: 0.6721
Epoch 5/5 accuracy: 0.5539 - loss: 0.6699 - val_accuracy: 0.5494 - val_loss: 0.6624
```
#### Tests End:
#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sequence_classification.py --rnn=SimpleRNN --epochs=5`
```
Epoch 1/5 accuracy: 0.4984 - loss: 0.7004 - val_accuracy: 0.5223 - val_loss: 0.6884
Epoch 2/5 accuracy: 0.5198 - loss: 0.6862 - val_accuracy: 0.5117 - val_loss: 0.6794
Epoch 3/5 accuracy: 0.5132 - loss: 0.6784 - val_accuracy: 0.5121 - val_loss: 0.6732
Epoch 4/5 accuracy: 0.5160 - loss: 0.6723 - val_accuracy: 0.5191 - val_loss: 0.6683
Epoch 5/5 accuracy: 0.5235 - loss: 0.6680 - val_accuracy: 0.5276 - val_loss: 0.6639
```

- `python3 sequence_classification.py --rnn=GRU --epochs=5`
```
Epoch 1/5 accuracy: 0.5109 - loss: 0.6929 - val_accuracy: 0.5128 - val_loss: 0.6915
Epoch 2/5 accuracy: 0.5174 - loss: 0.6894 - val_accuracy: 0.5155 - val_loss: 0.6785
Epoch 3/5 accuracy: 0.5446 - loss: 0.6630 - val_accuracy: 0.9538 - val_loss: 0.2142
Epoch 4/5 accuracy: 0.9812 - loss: 0.1270 - val_accuracy: 0.9987 - val_loss: 0.0304
Epoch 5/5 accuracy: 0.9985 - loss: 0.0270 - val_accuracy: 0.9995 - val_loss: 0.0135
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5`
```
Epoch 1/5 accuracy: 0.5131 - loss: 0.6930 - val_accuracy: 0.5187 - val_loss: 0.6918
Epoch 2/5 accuracy: 0.5187 - loss: 0.6892 - val_accuracy: 0.5340 - val_loss: 0.6760
Epoch 3/5 accuracy: 0.6401 - loss: 0.5744 - val_accuracy: 1.0000 - val_loss: 0.0845
Epoch 4/5 accuracy: 1.0000 - loss: 0.0585 - val_accuracy: 1.0000 - val_loss: 0.0194
Epoch 5/5 accuracy: 1.0000 - loss: 0.0154 - val_accuracy: 1.0000 - val_loss: 0.0082
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --hidden_layer=85`
```
Epoch 1/5 accuracy: 0.5151 - loss: 0.6888 - val_accuracy: 0.5323 - val_loss: 0.6571
Epoch 2/5 accuracy: 0.5387 - loss: 0.6497 - val_accuracy: 0.5575 - val_loss: 0.6321
Epoch 3/5 accuracy: 0.5570 - loss: 0.6242 - val_accuracy: 0.6199 - val_loss: 0.5854
Epoch 4/5 accuracy: 0.8367 - loss: 0.2854 - val_accuracy: 0.9897 - val_loss: 0.0503
Epoch 5/5 accuracy: 0.9995 - loss: 0.0058 - val_accuracy: 0.9999 - val_loss: 0.0014
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --hidden_layer=85 --clip_gradient=1`
```
Epoch 1/5 accuracy: 0.5151 - loss: 0.6888 - val_accuracy: 0.5323 - val_loss: 0.6571
Epoch 2/5 accuracy: 0.5387 - loss: 0.6497 - val_accuracy: 0.5582 - val_loss: 0.6321
Epoch 3/5 accuracy: 0.5576 - loss: 0.6237 - val_accuracy: 0.6542 - val_loss: 0.5625
Epoch 4/5 accuracy: 0.9033 - loss: 0.1909 - val_accuracy: 0.9999 - val_loss: 0.0014
Epoch 5/5 accuracy: 0.9997 - loss: 0.0029 - val_accuracy: 1.0000 - val_loss: 4.4711e-04
```
#### Examples End:
