### Assignment: sequence_classification
#### Date: Deadline: Apr 15, 22:00
#### Points: 2 points
#### Tests: sequence_classification_tests
#### Examples: sequence_classification_examples

The goal of this assignment is to introduce recurrent neural networks, show
their convergence speed, and illustrate exploding gradient issue. The network
should process sequences of 50 small integers and compute parity for each prefix
of the sequence. The inputs are either 0/1, or vectors with one-hot
representation of small integer.

Your goal is to modify the
[sequence_classification.py](https://github.com/ufal/npfl138/tree/master/labs/07/sequence_classification.py)
template and implement the following:
- Use the specified RNN type (`RNN`, `GRU`, and `LSTM`) and dimensionality.
- Process the sequence using the required RNN.
- Use additional hidden layer on the RNN outputs if requested.
- Implement gradient clipping if requested.

In addition to submitting the task in ReCodEx, please also run the following
variations and observe the results in TensorBoard.
Concentrate on the way how the RNNs converge, convergence speed, exploding
gradient issues and how gradient clipping helps:
- `--rnn=RNN --sequence_dim=1`, `--rnn=GRU --sequence_dim=1`, `--rnn=LSTM --sequence_dim=1`
- the same as above but with `--sequence_dim=3`
- the same as above but with `--sequence_dim=10`
- `--rnn=RNN --hidden_layer=75 --rnn_dim=30 --sequence_dim=30` and the same with `--clip_gradient=1`
- the same as above but with `--rnn=GRU` with and without `--clip_gradient=1`
- the same as above but with `--rnn=LSTM` with and without `--clip_gradient=1`

#### Tests Start: sequence_classification_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sequence_classification.py --recodex --train_sequences=1000 --sequence_length=20 --rnn=RNN --epochs=5`
```
Epoch 1/5 0.1s loss=0.7499 accuracy=0.5227 dev:loss=0.7166 dev:accuracy=0.5217
Epoch 2/5 0.1s loss=0.7017 accuracy=0.5203 dev:loss=0.6937 dev:accuracy=0.5217
Epoch 3/5 0.1s loss=0.6906 accuracy=0.5203 dev:loss=0.6880 dev:accuracy=0.5217
Epoch 4/5 0.1s loss=0.6868 accuracy=0.5203 dev:loss=0.6854 dev:accuracy=0.5217
Epoch 5/5 0.1s loss=0.6848 accuracy=0.5203 dev:loss=0.6840 dev:accuracy=0.5217
```

2. `python3 sequence_classification.py --recodex --train_sequences=1000 --sequence_length=20 --rnn=GRU --epochs=5`
```
Epoch 1/5 0.2s loss=0.6939 accuracy=0.4836 dev:loss=0.6931 dev:accuracy=0.5493
Epoch 2/5 0.2s loss=0.6928 accuracy=0.5231 dev:loss=0.6927 dev:accuracy=0.5397
Epoch 3/5 0.2s loss=0.6924 accuracy=0.5286 dev:loss=0.6921 dev:accuracy=0.5355
Epoch 4/5 0.2s loss=0.6920 accuracy=0.5294 dev:loss=0.6917 dev:accuracy=0.5140
Epoch 5/5 0.2s loss=0.6916 accuracy=0.5250 dev:loss=0.6913 dev:accuracy=0.5217
```

3. `python3 sequence_classification.py --recodex --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5`
```
Epoch 1/5 0.1s loss=0.6937 accuracy=0.4756 dev:loss=0.6935 dev:accuracy=0.4597
Epoch 2/5 0.1s loss=0.6931 accuracy=0.4979 dev:loss=0.6940 dev:accuracy=0.5027
Epoch 3/5 0.1s loss=0.6931 accuracy=0.5114 dev:loss=0.6937 dev:accuracy=0.5087
Epoch 4/5 0.1s loss=0.6929 accuracy=0.5192 dev:loss=0.6930 dev:accuracy=0.5296
Epoch 5/5 0.1s loss=0.6928 accuracy=0.5304 dev:loss=0.6929 dev:accuracy=0.5228
```

4. `python3 sequence_classification.py --recodex --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --rnn_dim=16 --hidden_layer=50 --sequence_dim=3`
```
Epoch 1/5 0.2s loss=0.6929 accuracy=0.5048 dev:loss=0.6914 dev:accuracy=0.5003
Epoch 2/5 0.1s loss=0.6901 accuracy=0.5192 dev:loss=0.6890 dev:accuracy=0.5149
Epoch 3/5 0.1s loss=0.6878 accuracy=0.5151 dev:loss=0.6858 dev:accuracy=0.5193
Epoch 4/5 0.1s loss=0.6833 accuracy=0.5161 dev:loss=0.6787 dev:accuracy=0.5156
Epoch 5/5 0.1s loss=0.6702 accuracy=0.5281 dev:loss=0.6595 dev:accuracy=0.5398
```

5. `python3 sequence_classification.py --recodex --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --rnn_dim=16 --hidden_layer=50 --sequence_dim=3 --clip_gradient=0.01`
```
Epoch 1/5 0.2s loss=0.6929 accuracy=0.4999 dev:loss=0.6903 dev:accuracy=0.5113
Epoch 2/5 0.2s loss=0.6894 accuracy=0.5148 dev:loss=0.6876 dev:accuracy=0.5185
Epoch 3/5 0.2s loss=0.6860 accuracy=0.5144 dev:loss=0.6833 dev:accuracy=0.5152
Epoch 4/5 0.2s loss=0.6795 accuracy=0.5156 dev:loss=0.6741 dev:accuracy=0.5189
Epoch 5/5 0.2s loss=0.6643 accuracy=0.5335 dev:loss=0.6534 dev:accuracy=0.5590
```
#### Tests End:
#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sequence_classification.py --rnn=RNN --epochs=5`
```
Epoch 1/5 1.6s loss=0.6938 accuracy=0.5073 dev:loss=0.6919 dev:accuracy=0.5121
Epoch 2/5 1.6s loss=0.6905 accuracy=0.5124 dev:loss=0.6885 dev:accuracy=0.5098
Epoch 3/5 1.6s loss=0.6854 accuracy=0.5122 dev:loss=0.6805 dev:accuracy=0.5167
Epoch 4/5 1.6s loss=0.6750 accuracy=0.5149 dev:loss=0.6706 dev:accuracy=0.5136
Epoch 5/5 1.6s loss=0.6688 accuracy=0.5148 dev:loss=0.6662 dev:accuracy=0.5129
```

- `python3 sequence_classification.py --rnn=GRU --epochs=5`
```
Epoch 1/5 3.6s loss=0.6929 accuracy=0.5061 dev:loss=0.6920 dev:accuracy=0.5157
Epoch 2/5 3.6s loss=0.6872 accuracy=0.5183 dev:loss=0.6767 dev:accuracy=0.5343
Epoch 3/5 3.6s loss=0.2817 accuracy=0.8612 dev:loss=0.0412 dev:accuracy=1.0000
Epoch 4/5 3.7s loss=0.0271 accuracy=0.9998 dev:loss=0.0180 dev:accuracy=1.0000
Epoch 5/5 3.7s loss=0.0144 accuracy=0.9998 dev:loss=0.0104 dev:accuracy=1.0000
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5`
```
Epoch 1/5 1.0s loss=0.6932 accuracy=0.5081 dev:loss=0.6929 dev:accuracy=0.5136
Epoch 2/5 0.9s loss=0.6927 accuracy=0.5135 dev:loss=0.6929 dev:accuracy=0.5142
Epoch 3/5 0.9s loss=0.6920 accuracy=0.5124 dev:loss=0.6914 dev:accuracy=0.5146
Epoch 4/5 0.9s loss=0.6903 accuracy=0.5108 dev:loss=0.6883 dev:accuracy=0.5084
Epoch 5/5 0.9s loss=0.6816 accuracy=0.5217 dev:loss=0.6724 dev:accuracy=0.5383
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --rnn_dim=30 --hidden_layer=75 --sequence_dim=30`
```
Epoch 1/5 1.7s loss=0.6931 accuracy=0.5053 dev:loss=0.6928 dev:accuracy=0.5089
Epoch 2/5 1.7s loss=0.6830 accuracy=0.5165 dev:loss=0.6540 dev:accuracy=0.5399
Epoch 3/5 1.7s loss=0.6261 accuracy=0.5555 dev:loss=0.5964 dev:accuracy=0.5797
Epoch 4/5 1.7s loss=0.5761 accuracy=0.5900 dev:loss=0.5547 dev:accuracy=0.6158
Epoch 5/5 1.7s loss=0.5423 accuracy=0.6549 dev:loss=0.6601 dev:accuracy=0.5286
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --rnn_dim=30 --hidden_layer=75 --sequence_dim=30 --clip_gradient=1`
```
Epoch 1/5 1.8s loss=0.6931 accuracy=0.5053 dev:loss=0.6928 dev:accuracy=0.5089
Epoch 2/5 1.8s loss=0.6830 accuracy=0.5165 dev:loss=0.6540 dev:accuracy=0.5399
Epoch 3/5 1.8s loss=0.6252 accuracy=0.5562 dev:loss=0.5943 dev:accuracy=0.5789
Epoch 4/5 1.8s loss=0.5732 accuracy=0.5934 dev:loss=0.5512 dev:accuracy=0.6213
Epoch 5/5 1.8s loss=0.2076 accuracy=0.8720 dev:loss=0.0012 dev:accuracy=1.0000
```
#### Examples End:
