### Assignment: sgd_backpropagation
#### Date: Deadline: Mar 12, 22:00
#### Points: 3 points
#### Tests: sgd_backpropagation_tests
#### Examples: sgd_backpropagation_examples

In this exercise you will learn how to compute gradients using the so-called
**automatic differentiation**, which allows to automatically run backpropagation
algorithm for a given computation. You can read the [Automatic Differentiation
with torch.autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
if interested. After computing the gradient, you should then perform training by
running manually implemented minibatch stochastic gradient descent.

Starting with the
[sgd_backpropagation.py](https://github.com/ufal/npfl138/tree/master/labs/02/sgd_backpropagation.py)
template, you should:
- implement a neural network with a single _tanh_ hidden layer and
  categorical output layer;
- compute the crossentropy loss;
- use `.backward()` to automatically compute the gradient of the loss
  with respect to all variables;
- perform the SGD update.

#### Tests Start: sgd_backpropagation_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Test accuracy after epoch 2 is 93.15
```

2. `python3 sgd_backpropagation.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.64
Dev accuracy after epoch 2 is 94.80
Test accuracy after epoch 2 is 93.54
```
#### Tests End:
#### Examples Start: sgd_backpropagation_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sgd_backpropagation.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Dev accuracy after epoch 3 is 95.16
Dev accuracy after epoch 4 is 95.50
Dev accuracy after epoch 5 is 95.96
Dev accuracy after epoch 6 is 96.04
Dev accuracy after epoch 7 is 95.82
Dev accuracy after epoch 8 is 95.92
Dev accuracy after epoch 9 is 95.96
Dev accuracy after epoch 10 is 96.16
Test accuracy after epoch 10 is 95.26
```

- `python3 sgd_backpropagation.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.64
Dev accuracy after epoch 2 is 94.80
Dev accuracy after epoch 3 is 95.56
Dev accuracy after epoch 4 is 95.98
Dev accuracy after epoch 5 is 96.24
Dev accuracy after epoch 6 is 96.74
Dev accuracy after epoch 7 is 96.52
Dev accuracy after epoch 8 is 96.54
Dev accuracy after epoch 9 is 97.04
Dev accuracy after epoch 10 is 97.02
Test accuracy after epoch 10 is 96.16
```
#### Examples End:
