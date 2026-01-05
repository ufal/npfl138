### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 05, 22:00
#### Points: 2 points
#### Tests: mnist_layers_activations_tests
#### Examples: mnist_layers_activations_examples

Before solving the assignment, start by playing with
[example_pytorch_tensorboard.py](https://github.com/ufal/npfl138/tree/past-2425/labs/01/example_pytorch_tensorboard.py),
in order to familiarize with PyTorch and TensorBoard. After running the example,
start TensorBoard in the same directory using `tensorboard --logdir logs` and
open <http://localhost:6006> in a browser and explore the generated logs.

**Your goal** is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl138/tree/past-2425/labs/01/mnist_layers_activations.py)
template such that a user-specified neural network is constructed:
- A number of hidden layers (including zero) can be specified on the command line
  using the parameter `hidden_layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.

#### Tests Start: mnist_layers_activations_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=0 --activation=none`
```
Epoch 1/1 1.0s train_loss=0.5374 train_accuracy=0.8614 dev_loss=0.2768 dev_accuracy=0.9270
```

2. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=none`
```
Epoch 1/1 1.4s train_loss=0.3791 train_accuracy=0.8922 dev_loss=0.2400 dev_accuracy=0.9290
```

3. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=relu`
```
Epoch 1/1 1.5s train_loss=0.3178 train_accuracy=0.9104 dev_loss=0.1482 dev_accuracy=0.9566
```

4. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=tanh`
```
Epoch 1/1 1.4s train_loss=0.3318 train_accuracy=0.9061 dev_loss=0.1632 dev_accuracy=0.9530
```

5. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=1 --activation=sigmoid`
```
Epoch 1/1 1.4s train_loss=0.4985 train_accuracy=0.8788 dev_loss=0.2156 dev_accuracy=0.9382
```

6. `python3 mnist_layers_activations.py --epochs=1 --hidden_layers=3 --activation=relu`
```
Epoch 1/1 1.7s train_loss=0.2700 train_accuracy=0.9213 dev_loss=0.1188 dev_accuracy=0.9680
```
#### Tests End:
#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 1.0s train_loss=0.5374 train_accuracy=0.8614 dev_loss=0.2768 dev_accuracy=0.9270
Epoch  5/10 1.0s train_loss=0.2779 train_accuracy=0.9220 dev_loss=0.2201 dev_accuracy=0.9430
Epoch 10/10 1.0s train_loss=0.2591 train_accuracy=0.9278 dev_loss=0.2139 dev_accuracy=0.9432
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 1.4s train_loss=0.3791 train_accuracy=0.8922 dev_loss=0.2400 dev_accuracy=0.9290
Epoch  5/10 1.4s train_loss=0.2775 train_accuracy=0.9225 dev_loss=0.2217 dev_accuracy=0.9396
Epoch 10/10 1.4s train_loss=0.2645 train_accuracy=0.9247 dev_loss=0.2264 dev_accuracy=0.9378
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 1.4s train_loss=0.3178 train_accuracy=0.9104 dev_loss=0.1482 dev_accuracy=0.9566
Epoch  5/10 1.5s train_loss=0.0627 train_accuracy=0.9811 dev_loss=0.0827 dev_accuracy=0.9786
Epoch 10/10 1.6s train_loss=0.0240 train_accuracy=0.9930 dev_loss=0.0782 dev_accuracy=0.9810
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 1.4s train_loss=0.3318 train_accuracy=0.9061 dev_loss=0.1632 dev_accuracy=0.9530
Epoch  5/10 1.4s train_loss=0.0732 train_accuracy=0.9798 dev_loss=0.0837 dev_accuracy=0.9768
Epoch 10/10 1.5s train_loss=0.0254 train_accuracy=0.9943 dev_loss=0.0733 dev_accuracy=0.9790
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 1.4s train_loss=0.4985 train_accuracy=0.8788 dev_loss=0.2156 dev_accuracy=0.9382
Epoch  5/10 1.4s train_loss=0.1249 train_accuracy=0.9641 dev_loss=0.1077 dev_accuracy=0.9698
Epoch 10/10 1.4s train_loss=0.0605 train_accuracy=0.9837 dev_loss=0.0781 dev_accuracy=0.9762
```

- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 1.7s train_loss=0.2700 train_accuracy=0.9213 dev_loss=0.1188 dev_accuracy=0.9680
Epoch  5/10 1.9s train_loss=0.0477 train_accuracy=0.9849 dev_loss=0.0787 dev_accuracy=0.9794
Epoch 10/10 1.9s train_loss=0.0248 train_accuracy=0.9916 dev_loss=0.1015 dev_accuracy=0.9762
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 2.8s train_loss=0.3562 train_accuracy=0.8911 dev_loss=0.1556 dev_accuracy=0.9598
Epoch  5/10 3.3s train_loss=0.0864 train_accuracy=0.9764 dev_loss=0.1164 dev_accuracy=0.9686
Epoch 10/10 3.3s train_loss=0.0474 train_accuracy=0.9874 dev_loss=0.0877 dev_accuracy=0.9774
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 2.6s train_loss=1.9711 train_accuracy=0.1803 dev_loss=1.8477 dev_accuracy=0.2148
Epoch  5/10 2.6s train_loss=0.9947 train_accuracy=0.5815 dev_loss=0.8246 dev_accuracy=0.6392
Epoch 10/10 2.6s train_loss=0.4406 train_accuracy=0.8924 dev_loss=0.4239 dev_accuracy=0.8992
```
#### Examples End:
