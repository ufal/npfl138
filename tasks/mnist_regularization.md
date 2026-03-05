### Assignment: mnist_regularization
#### Date: Deadline: Mar 18, 22:00
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

1. `python3 mnist_regularization.py --recodex --epochs=1 --dropout=0.3`
```
Epoch 1/1 0.5s loss=0.8099 accuracy=0.7600 dev:loss=0.3311 dev:accuracy=0.9066
```

2. `python3 mnist_regularization.py --recodex --epochs=1 --dropout=0.5 --hidden_layers 300 300`
```
Epoch 1/1 0.6s loss=1.4732 accuracy=0.4886 dev:loss=0.4817 dev:accuracy=0.8772
```

3. `python3 mnist_regularization.py --recodex --epochs=1 --weight_decay=0.1`
```
Epoch 1/1 0.5s loss=0.6149 accuracy=0.8308 dev:loss=0.2904 dev:accuracy=0.9162
```

4. `python3 mnist_regularization.py --recodex --epochs=1 --weight_decay=0.3`
```
Epoch 1/1 0.5s loss=0.6169 accuracy=0.8302 dev:loss=0.2924 dev:accuracy=0.9160
```

5. `python3 mnist_regularization.py --recodex --epochs=1 --label_smoothing=0.1`
```
Epoch 1/1 0.5s loss=0.9985 accuracy=0.8354 dev:loss=0.7706 dev:accuracy=0.9214
```

6. `python3 mnist_regularization.py --recodex --epochs=1 --label_smoothing=0.3`
```
Epoch 1/1 0.5s loss=1.5091 accuracy=0.8382 dev:loss=1.3680 dev:accuracy=0.9298
```
#### Tests End:
