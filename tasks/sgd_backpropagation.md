### Assignment: sgd_backpropagation
#### Date: Deadline: Mar 12, 22:00
#### Points: 3 points
#### Tests: sgd_backpropagation_tests
#### Examples: sgd_backpropagation_examples

_**The template was updated on Feb 27, 17:30.** The original one did not shuffle
the training data. You do not need to redownload it, ReCodEx accepts both variants.
However, the Tests and Examples have been regenerated using the updated template._

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
Dev accuracy after epoch 1 is 92.98
Dev accuracy after epoch 2 is 94.42
Test accuracy after epoch 2 is 92.72
```

2. `python3 sgd_backpropagation.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.58
Dev accuracy after epoch 2 is 95.26
Test accuracy after epoch 2 is 93.75
```
#### Tests End:
#### Examples Start: sgd_backpropagation_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sgd_backpropagation.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.98
Dev accuracy after epoch 2 is 94.42
Dev accuracy after epoch 3 is 94.68
Dev accuracy after epoch 4 is 95.08
Dev accuracy after epoch 5 is 95.28
Dev accuracy after epoch 6 is 95.20
Dev accuracy after epoch 7 is 95.52
Dev accuracy after epoch 8 is 95.32
Dev accuracy after epoch 9 is 95.66
Dev accuracy after epoch 10 is 95.84
Test accuracy after epoch 10 is 95.02
```

- `python3 sgd_backpropagation.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.58
Dev accuracy after epoch 2 is 95.26
Dev accuracy after epoch 3 is 95.66
Dev accuracy after epoch 4 is 95.90
Dev accuracy after epoch 5 is 96.26
Dev accuracy after epoch 6 is 96.52
Dev accuracy after epoch 7 is 96.52
Dev accuracy after epoch 8 is 96.74
Dev accuracy after epoch 9 is 96.74
Dev accuracy after epoch 10 is 96.62
Test accuracy after epoch 10 is 95.84
```
#### Examples End:
