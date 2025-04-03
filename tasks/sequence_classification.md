### Assignment: sequence_classification
#### Date: Deadline: Apr 16, 22:00
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

1. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=RNN --epochs=5`
```
Epoch 1/5 0.1s train_loss=0.7008 train_accuracy=0.4762 dev_loss=0.6952 dev_accuracy=0.4778
Epoch 2/5 0.1s train_loss=0.6938 train_accuracy=0.5011 dev_loss=0.6926 dev_accuracy=0.4979
Epoch 3/5 0.1s train_loss=0.6922 train_accuracy=0.5215 dev_loss=0.6918 dev_accuracy=0.5395
Epoch 4/5 0.1s train_loss=0.6913 train_accuracy=0.5423 dev_loss=0.6912 dev_accuracy=0.5362
Epoch 5/5 0.1s train_loss=0.6909 train_accuracy=0.5508 dev_loss=0.6909 dev_accuracy=0.5405
```

2. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=GRU --epochs=5`
```
Epoch 1/5 0.2s train_loss=0.6960 train_accuracy=0.4760 dev_loss=0.6943 dev_accuracy=0.4747
Epoch 2/5 0.2s train_loss=0.6936 train_accuracy=0.4967 dev_loss=0.6931 dev_accuracy=0.5026
Epoch 3/5 0.2s train_loss=0.6926 train_accuracy=0.5062 dev_loss=0.6924 dev_accuracy=0.5296
Epoch 4/5 0.2s train_loss=0.6920 train_accuracy=0.5307 dev_loss=0.6919 dev_accuracy=0.5267
Epoch 5/5 0.2s train_loss=0.6917 train_accuracy=0.5307 dev_loss=0.6915 dev_accuracy=0.5321
```

3. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5`
```
Epoch 1/5 0.1s train_loss=0.6939 train_accuracy=0.4993 dev_loss=0.6931 dev_accuracy=0.5065
Epoch 2/5 0.1s train_loss=0.6932 train_accuracy=0.5007 dev_loss=0.6931 dev_accuracy=0.5027
Epoch 3/5 0.1s train_loss=0.6929 train_accuracy=0.5115 dev_loss=0.6928 dev_accuracy=0.5483
Epoch 4/5 0.1s train_loss=0.6927 train_accuracy=0.5444 dev_loss=0.6928 dev_accuracy=0.5480
Epoch 5/5 0.1s train_loss=0.6925 train_accuracy=0.5403 dev_loss=0.6931 dev_accuracy=0.5407
```

4. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --rnn_dim=16 --hidden_layer=50 --sequence_dim=3`
```
Epoch 1/5 0.1s train_loss=0.6928 train_accuracy=0.4956 dev_loss=0.6903 dev_accuracy=0.5160
Epoch 2/5 0.1s train_loss=0.6886 train_accuracy=0.5182 dev_loss=0.6892 dev_accuracy=0.5152
Epoch 3/5 0.1s train_loss=0.6835 train_accuracy=0.5138 dev_loss=0.6785 dev_accuracy=0.5124
Epoch 4/5 0.1s train_loss=0.6691 train_accuracy=0.5493 dev_loss=0.6596 dev_accuracy=0.5347
Epoch 5/5 0.1s train_loss=0.6474 train_accuracy=0.5756 dev_loss=0.6342 dev_accuracy=0.5939
```

5. `python3 sequence_classification.py --train_sequences=1000 --sequence_length=20 --rnn=LSTM --epochs=5 --rnn_dim=16 --hidden_layer=50 --sequence_dim=3 --clip_gradient=0.01`
```
Epoch 1/5 0.1s train_loss=0.6928 train_accuracy=0.4946 dev_loss=0.6900 dev_accuracy=0.5055
Epoch 2/5 0.1s train_loss=0.6882 train_accuracy=0.5167 dev_loss=0.6871 dev_accuracy=0.5135
Epoch 3/5 0.1s train_loss=0.6814 train_accuracy=0.5083 dev_loss=0.6756 dev_accuracy=0.5107
Epoch 4/5 0.1s train_loss=0.6684 train_accuracy=0.5483 dev_loss=0.6609 dev_accuracy=0.5204
Epoch 5/5 0.1s train_loss=0.6504 train_accuracy=0.5754 dev_loss=0.6404 dev_accuracy=0.5936
```
#### Tests End:
#### Examples Start: sequence_classification_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sequence_classification.py --rnn=RNN --epochs=5`
```
Epoch 1/5 1.3s train_loss=0.6938 train_accuracy=0.5073 dev_loss=0.6919 dev_accuracy=0.5121
Epoch 2/5 1.3s train_loss=0.6905 train_accuracy=0.5124 dev_loss=0.6885 dev_accuracy=0.5098
Epoch 3/5 1.3s train_loss=0.6854 train_accuracy=0.5122 dev_loss=0.6805 dev_accuracy=0.5167
Epoch 4/5 1.3s train_loss=0.6750 train_accuracy=0.5149 dev_loss=0.6706 dev_accuracy=0.5136
Epoch 5/5 1.3s train_loss=0.6688 train_accuracy=0.5148 dev_loss=0.6662 dev_accuracy=0.5129
```

- `python3 sequence_classification.py --rnn=GRU --epochs=5`
```
Epoch 1/5 2.9s train_loss=0.6929 train_accuracy=0.5061 dev_loss=0.6920 dev_accuracy=0.5157
Epoch 2/5 2.8s train_loss=0.6872 train_accuracy=0.5183 dev_loss=0.6767 dev_accuracy=0.5343
Epoch 3/5 2.9s train_loss=0.2817 train_accuracy=0.8612 dev_loss=0.0412 dev_accuracy=1.0000
Epoch 4/5 2.9s train_loss=0.0271 train_accuracy=0.9998 dev_loss=0.0180 dev_accuracy=1.0000
Epoch 5/5 2.9s train_loss=0.0144 train_accuracy=0.9998 dev_loss=0.0104 dev_accuracy=1.0000
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5`
```
Epoch 1/5 0.8s train_loss=0.6932 train_accuracy=0.5081 dev_loss=0.6929 dev_accuracy=0.5136
Epoch 2/5 0.8s train_loss=0.6927 train_accuracy=0.5135 dev_loss=0.6929 dev_accuracy=0.5142
Epoch 3/5 0.8s train_loss=0.6920 train_accuracy=0.5124 dev_loss=0.6914 dev_accuracy=0.5146
Epoch 4/5 0.8s train_loss=0.6903 train_accuracy=0.5108 dev_loss=0.6883 dev_accuracy=0.5084
Epoch 5/5 0.8s train_loss=0.6816 train_accuracy=0.5217 dev_loss=0.6724 dev_accuracy=0.5383
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --rnn_dim=30 --hidden_layer=75 --sequence_dim=30`
```
Epoch 1/5 1.3s train_loss=0.6931 train_accuracy=0.5053 dev_loss=0.6928 dev_accuracy=0.5089
Epoch 2/5 1.3s train_loss=0.6830 train_accuracy=0.5165 dev_loss=0.6540 dev_accuracy=0.5399
Epoch 3/5 1.3s train_loss=0.6261 train_accuracy=0.5555 dev_loss=0.5964 dev_accuracy=0.5797
Epoch 4/5 1.3s train_loss=0.5761 train_accuracy=0.5900 dev_loss=0.5547 dev_accuracy=0.6158
Epoch 5/5 1.3s train_loss=0.5423 train_accuracy=0.6549 dev_loss=0.6601 dev_accuracy=0.5286
```

- `python3 sequence_classification.py --rnn=LSTM --epochs=5 --rnn_dim=30 --hidden_layer=75 --sequence_dim=30 --clip_gradient=1`
```
Epoch 1/5 1.4s train_loss=0.6931 train_accuracy=0.5053 dev_loss=0.6928 dev_accuracy=0.5089
Epoch 2/5 1.4s train_loss=0.6830 train_accuracy=0.5165 dev_loss=0.6540 dev_accuracy=0.5399
Epoch 3/5 1.4s train_loss=0.6252 train_accuracy=0.5562 dev_loss=0.5943 dev_accuracy=0.5789
Epoch 4/5 1.4s train_loss=0.5732 train_accuracy=0.5932 dev_loss=0.5512 dev_accuracy=0.6212
Epoch 5/5 1.4s train_loss=0.2098 train_accuracy=0.8723 dev_loss=0.0015 dev_accuracy=1.0000
```
#### Examples End:
