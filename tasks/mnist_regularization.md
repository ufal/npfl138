### Assignment: mnist_regularization
#### Date: Deadline: Mar 19, 22:00
#### Points: 3 points
#### Tests: mnist_regularization_tests

You will learn how to implement three regularization methods in this assignment.
Start with the
[mnist_regularization.py](https://github.com/ufal/npfl138/tree/master/labs/03/mnist_regularization.py)
template and implement the following:
- Allow using dropout with rate `args.dropout`. Add a dropout layer after the
  first `Flatten` and also after all `Dense` hidden layers (but not after the
  output layer).
- Allow using AdamW with weight decay with strength of `args.weight_decay`,
  making sure the weight decay is not applied on bias.
- Allow using label smoothing with weight `args.label_smoothing`. Instead
  of `SparseCategoricalCrossentropy`, you will need to use
  `CategoricalCrossentropy` which offers `label_smoothing` argument.

In addition to submitting the task in ReCodEx, also run the following
variations and observe the results in TensorBoard,
notably the training, development and test set accuracy and loss:
- dropout rate `0`, `0.3`, `0.5`, `0.6`, `0.8`;
- weight decay `0`, `0.1`, `0.3`, `0.5`, `0.1`;
- label smoothing `0`, `0.1`, `0.3`, `0.5`.

#### Tests Start: mnist_regularization_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_regularization.py --epochs=1 --dropout=0.3`
```
accuracy: 0.5981 - loss: 1.2688 - val_accuracy: 0.9174 - val_loss: 0.3051
```

2. `python3 mnist_regularization.py --epochs=1 --dropout=0.5 --hidden_layers 300 300`
```
accuracy: 0.3429 - loss: 1.9163 - val_accuracy: 0.8826 - val_loss: 0.4937
```

3. `python3 mnist_regularization.py --epochs=1 --weight_decay=0.1`
```
accuracy: 0.7014 - loss: 1.0412 - val_accuracy: 0.9236 - val_loss: 0.2776
```

4. `python3 mnist_regularization.py --epochs=1 --weight_decay=0.3`
```
accuracy: 0.7006 - loss: 1.0429 - val_accuracy: 0.9232 - val_loss: 0.2801
```

5. `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.1`
```
accuracy: 0.7102 - loss: 1.3015 - val_accuracy: 0.9276 - val_loss: 0.7656
```

6. `python3 mnist_regularization.py --epochs=1 --label_smoothing=0.3`
```
accuracy: 0.7113 - loss: 1.6854 - val_accuracy: 0.9332 - val_loss: 1.3709
```
#### Tests End:
