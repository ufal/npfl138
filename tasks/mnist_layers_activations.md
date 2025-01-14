### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 05, 22:00
#### Points: 2 points
#### Tests: mnist_layers_activations_tests
#### Examples: mnist_layers_activations_examples

Before solving the assignment, start by playing with
[example_keras_tensorboard.py](https://github.com/ufal/npfl138/tree/past-2324/labs/01/example_keras_tensorboard.py),
in order to familiarize with Keras and TensorBoard.
Run it, and when it finishes, run TensorBoard using `tensorboard --logdir logs`.
Then open <http://localhost:6006> in a browser and explore the active tabs.

**Your goal** is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl138/tree/past-2324/labs/01/mnist_layers_activations.py)
template such that a user-specified neural network is constructed:
- A number of hidden layers (including zero) can be specified on the command line
  using parameter `hidden_layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.

#### Tests Start: mnist_layers_activations_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=0 --activation=none`
```
accuracy: 0.7801 - loss: 0.8405 - val_accuracy: 0.9300 - val_loss: 0.2716
```

2. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=none`
```
accuracy: 0.8483 - loss: 0.5230 - val_accuracy: 0.9352 - val_loss: 0.2422
```

3. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=relu`
```
accuracy: 0.8503 - loss: 0.5286 - val_accuracy: 0.9604 - val_loss: 0.1432
```

4. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=tanh`
```
accuracy: 0.8529 - loss: 0.5183 - val_accuracy: 0.9564 - val_loss: 0.1632
```

5. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=sigmoid`
```
accuracy: 0.7851 - loss: 0.8650 - val_accuracy: 0.9414 - val_loss: 0.2196
```

6. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=3 --activation=relu`
```
accuracy: 0.8497 - loss: 0.5011 - val_accuracy: 0.9664 - val_loss: 0.1225
```
#### Tests End:
#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 accuracy: 0.7801 - loss: 0.8405 - val_accuracy: 0.9300 - val_loss: 0.2716
Epoch  5/10 accuracy: 0.9222 - loss: 0.2792 - val_accuracy: 0.9406 - val_loss: 0.2203
Epoch 10/10 accuracy: 0.9304 - loss: 0.2515 - val_accuracy: 0.9432 - val_loss: 0.2159
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 accuracy: 0.8483 - loss: 0.5230 - val_accuracy: 0.9352 - val_loss: 0.2422
Epoch  5/10 accuracy: 0.9236 - loss: 0.2758 - val_accuracy: 0.9360 - val_loss: 0.2325
Epoch 10/10 accuracy: 0.9298 - loss: 0.2517 - val_accuracy: 0.9354 - val_loss: 0.2439
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 accuracy: 0.8503 - loss: 0.5286 - val_accuracy: 0.9604 - val_loss: 0.1432
Epoch  5/10 accuracy: 0.9824 - loss: 0.0613 - val_accuracy: 0.9808 - val_loss: 0.0740
Epoch 10/10 accuracy: 0.9948 - loss: 0.0202 - val_accuracy: 0.9788 - val_loss: 0.0821
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 accuracy: 0.8529 - loss: 0.5183 - val_accuracy: 0.9564 - val_loss: 0.1632
Epoch  5/10 accuracy: 0.9800 - loss: 0.0728 - val_accuracy: 0.9740 - val_loss: 0.0853
Epoch 10/10 accuracy: 0.9948 - loss: 0.0244 - val_accuracy: 0.9782 - val_loss: 0.0772
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 accuracy: 0.7851 - loss: 0.8650 - val_accuracy: 0.9414 - val_loss: 0.2196
Epoch  5/10 accuracy: 0.9647 - loss: 0.1270 - val_accuracy: 0.9704 - val_loss: 0.1079
Epoch 10/10 accuracy: 0.9852 - loss: 0.0583 - val_accuracy: 0.9756 - val_loss: 0.0837
```

- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 accuracy: 0.8497 - loss: 0.5011 - val_accuracy: 0.9664 - val_loss: 0.1225
Epoch  5/10 accuracy: 0.9862 - loss: 0.0438 - val_accuracy: 0.9734 - val_loss: 0.1026
Epoch 10/10 accuracy: 0.9932 - loss: 0.0202 - val_accuracy: 0.9818 - val_loss: 0.0865
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 accuracy: 0.7710 - loss: 0.6793 - val_accuracy: 0.9570 - val_loss: 0.1479
Epoch  5/10 accuracy: 0.9780 - loss: 0.0783 - val_accuracy: 0.9786 - val_loss: 0.0808
Epoch 10/10 accuracy: 0.9869 - loss: 0.0481 - val_accuracy: 0.9724 - val_loss: 0.1163
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 accuracy: 0.1072 - loss: 2.3068 - val_accuracy: 0.1784 - val_loss: 2.1247
Epoch  5/10 accuracy: 0.8825 - loss: 0.4776 - val_accuracy: 0.9164 - val_loss: 0.3686
Epoch 10/10 accuracy: 0.9294 - loss: 0.2994 - val_accuracy: 0.9386 - val_loss: 0.2671
```
#### Examples End:
