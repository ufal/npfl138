### Assignment: mnist_regularization
#### Date: Deadline: Mar 19, 22:00
#### Points: 3 points
#### Tests: mnist_regularization_tests

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl138/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Linear` hidden layers (but not after the
  output layer).
- Allow using AdamW with weight decay with strength of `args.weight_decay`,
  making sure the weight decay is not applied on bias.
- Allow using label smoothing with weight `args.label_smoothing`.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard,
notably the training, development and test set accuracy and loss:
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- weight decay `0`, `0.1`, `0.3`, `0.5`, `1.0`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Tests Start: mnist_regularization_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_regularization.py --epochs=1 --dropout=0.3`
```
Epoch 1/1 0.4s train_loss=0.7775 train_accuracy=0.7704 dev_loss=0.3211 dev_accuracy=0.9122
```

2. `python3 mnist_regularization.py --epochs=1 --dropout=0.5 --hidden_layers 300 300`
```
Epoch 1/1 0.4s train_loss=1.5365 train_accuracy=0.4824 dev_loss=0.5010 dev_accuracy=0.8680
```

3. `python3 mnist_regularization.py --epochs=1 --weight_decay=0.1`
```
Epoch 1/1 0.4s train_loss=0.5948 train_accuracy=0.8386 dev_loss=0.2868 dev_accuracy=0.9206
```

4. `python3 mnist_regularization.py --epochs=1 --weight_decay=0.3`
```
Epoch 1/1 0.4s train_loss=0.5969 train_accuracy=0.8386 dev_loss=0.2890 dev_accuracy=0.9206
```

5. `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.1`
```
Epoch 1/1 0.4s train_loss=0.9841 train_accuracy=0.8442 dev_loss=0.7734 dev_accuracy=0.9244
```

6. `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.3`
```
Epoch 1/1 0.4s train_loss=1.5040 train_accuracy=0.8458 dev_loss=1.3727 dev_accuracy=0.9312
```
#### Tests End:
