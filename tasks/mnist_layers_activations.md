### Assignment: mnist_layers_activations
#### Date: Deadline: Mar 04, 22:00
#### Points: 2 points
#### Tests: mnist_layers_activations_tests
#### Examples: mnist_layers_activations_examples

Before solving the assignment, start by playing with
[example_pytorch_tensorboard.py](https://github.com/ufal/npfl138/tree/master/labs/01/example_pytorch_tensorboard.py),
in order to familiarize with PyTorch and TensorBoard. After running the example,
start TensorBoard in the same directory using `tensorboard --logdir logs` and
open <http://localhost:6006> in a browser and explore the generated logs.

**Your goal** is to modify the
[mnist_layers_activations.py](https://github.com/ufal/npfl138/tree/master/labs/01/mnist_layers_activations.py)
template such that a user-specified neural network is constructed:
- A number of hidden layers (including zero) can be specified on the command line
  using the parameter `hidden_layers`.
- Activation function of these hidden layers can be also specified as a command
  line parameter `activation`, with supported values of `none`, `relu`, `tanh`
  and `sigmoid`.

#### Tests Start: mnist_layers_activations_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=0 --activation=none`
```
Epoch 1/1 1.1s loss=0.5340 accuracy=0.8632 dev:loss=0.2762 dev:accuracy=0.9278
```

2. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=1 --activation=none`
```
Epoch 1/1 1.6s loss=0.3791 accuracy=0.8913 dev:loss=0.2372 dev:accuracy=0.9314
```

3. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=1 --activation=relu`
```
Epoch 1/1 1.7s loss=0.3149 accuracy=0.9110 dev:loss=0.1458 dev:accuracy=0.9608
```

4. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=1 --activation=tanh`
```
Epoch 1/1 1.7s loss=0.3333 accuracy=0.9049 dev:loss=0.1613 dev:accuracy=0.9582
```

5. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=1 --activation=sigmoid`
```
Epoch 1/1 1.7s loss=0.4900 accuracy=0.8782 dev:loss=0.2185 dev:accuracy=0.9390
```

6. `python3 mnist_layers_activations.py --recodex --epochs=1 --hidden_layers=3 --activation=relu`
```
Epoch 1/1 2.1s loss=0.2736 accuracy=0.9194 dev:loss=0.1089 dev:accuracy=0.9676
```
#### Tests End:
#### Examples Start: mnist_layers_activations_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_layers_activations.py --hidden_layers=0 --activation=none`
```
Epoch  1/10 1.1s loss=0.5374 accuracy=0.8614 dev:loss=0.2768 dev:accuracy=0.9270
Epoch  5/10 1.1s loss=0.2779 accuracy=0.9220 dev:loss=0.2201 dev:accuracy=0.9430
Epoch 10/10 1.1s loss=0.2591 accuracy=0.9278 dev:loss=0.2139 dev:accuracy=0.9432
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=none`
```
Epoch  1/10 1.7s loss=0.3791 accuracy=0.8922 dev:loss=0.2400 dev:accuracy=0.9290
Epoch  5/10 1.7s loss=0.2775 accuracy=0.9225 dev:loss=0.2217 dev:accuracy=0.9396
Epoch 10/10 1.7s loss=0.2645 accuracy=0.9247 dev:loss=0.2264 dev:accuracy=0.9378
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=relu`
```
Epoch  1/10 1.7s loss=0.3178 accuracy=0.9104 dev:loss=0.1482 dev:accuracy=0.9566
Epoch  5/10 1.9s loss=0.0627 accuracy=0.9811 dev:loss=0.0827 dev:accuracy=0.9786
Epoch 10/10 1.9s loss=0.0240 accuracy=0.9930 dev:loss=0.0782 dev:accuracy=0.9810
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=tanh`
```
Epoch  1/10 1.7s loss=0.3318 accuracy=0.9061 dev:loss=0.1632 dev:accuracy=0.9530
Epoch  5/10 1.7s loss=0.0732 accuracy=0.9798 dev:loss=0.0837 dev:accuracy=0.9768
Epoch 10/10 1.8s loss=0.0254 accuracy=0.9943 dev:loss=0.0733 dev:accuracy=0.9790
```

- `python3 mnist_layers_activations.py --hidden_layers=1 --activation=sigmoid`
```
Epoch  1/10 1.7s loss=0.4985 accuracy=0.8788 dev:loss=0.2156 dev:accuracy=0.9382
Epoch  5/10 1.8s loss=0.1249 accuracy=0.9641 dev:loss=0.1077 dev:accuracy=0.9698
Epoch 10/10 1.8s loss=0.0605 accuracy=0.9837 dev:loss=0.0781 dev:accuracy=0.9762
```

- `python3 mnist_layers_activations.py --hidden_layers=3 --activation=relu`
```
Epoch  1/10 2.1s loss=0.2700 accuracy=0.9213 dev:loss=0.1188 dev:accuracy=0.9680
Epoch  5/10 2.2s loss=0.0477 accuracy=0.9849 dev:loss=0.0787 dev:accuracy=0.9794
Epoch 10/10 2.3s loss=0.0248 accuracy=0.9916 dev:loss=0.1015 dev:accuracy=0.9762
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=relu`
```
Epoch  1/10 3.4s loss=0.3562 accuracy=0.8911 dev:loss=0.1556 dev:accuracy=0.9598
Epoch  5/10 3.9s loss=0.0864 accuracy=0.9764 dev:loss=0.1164 dev:accuracy=0.9686
Epoch 10/10 4.0s loss=0.0474 accuracy=0.9874 dev:loss=0.0877 dev:accuracy=0.9774
```

- `python3 mnist_layers_activations.py --hidden_layers=10 --activation=sigmoid`
```
Epoch  1/10 3.1s loss=1.9711 accuracy=0.1803 dev:loss=1.8477 dev:accuracy=0.2148
Epoch  5/10 3.2s loss=0.9947 accuracy=0.5815 dev:loss=0.8246 dev:accuracy=0.6392
Epoch 10/10 3.2s loss=0.4406 accuracy=0.8924 dev:loss=0.4239 dev:accuracy=0.8992
```
#### Examples End:
