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

This assignment also demonstrate the most important parts of the
[npfl138.TrainableModule](https://github.com/ufal/npfl138/tree/master/labs/npfl138/trainable_module.py)
that we are using.

#### Tests Start: sgd_backpropagation_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sgd_backpropagation.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.92
Dev accuracy after epoch 2 is 94.08
Test accuracy after epoch 2 is 92.46
```

2. `python3 sgd_backpropagation.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.80
Dev accuracy after epoch 2 is 95.30
Test accuracy after epoch 2 is 93.63
```
#### Tests End:
#### Examples Start: sgd_backpropagation_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sgd_backpropagation.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.92
Dev accuracy after epoch 2 is 94.08
Dev accuracy after epoch 3 is 94.74
Dev accuracy after epoch 4 is 95.00
Dev accuracy after epoch 5 is 95.34
Dev accuracy after epoch 6 is 95.34
Dev accuracy after epoch 7 is 95.46
Dev accuracy after epoch 8 is 95.60
Dev accuracy after epoch 9 is 95.74
Dev accuracy after epoch 10 is 95.72
Test accuracy after epoch 10 is 94.67
```

- `python3 sgd_backpropagation.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.80
Dev accuracy after epoch 2 is 95.30
Dev accuracy after epoch 3 is 95.80
Dev accuracy after epoch 4 is 96.08
Dev accuracy after epoch 5 is 96.18
Dev accuracy after epoch 6 is 96.26
Dev accuracy after epoch 7 is 96.36
Dev accuracy after epoch 8 is 96.44
Dev accuracy after epoch 9 is 96.42
Dev accuracy after epoch 10 is 96.42
Test accuracy after epoch 10 is 95.78
```
#### Examples End:
