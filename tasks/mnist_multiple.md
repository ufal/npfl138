### Assignment: mnist_multiple
#### Date: Deadline: Mar 25, 22:00
#### Points: 3 points
#### Tests: mnist_multiple_tests

In this assignment you will implement a model with multiple inputs and outputs.
Start with the [mnist_multiple.py](https://github.com/ufal/npfl138/tree/master/labs/04/mnist_multiple.py)
template.

![mnist_multiple](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/tasks/figures/mnist_multiple.svgz)

- The goal is to create a model which given two input MNIST images determines if the
  digit on the first one is greater than on the second one.
- We perform this comparison in two different ways:
  - by directly predicting the comparison by the network (_direct comparison_), and
  - by first classifying the images into digits and then comparing these predictions (_indirect comparison_).
- The model has four outputs:
  - _direct comparison_ whether the first digit is greater than the second one,
  - digit classification for the first image,
  - digit classification for the second image,
  - _indirect comparison_ comparing the most probable digits predicted by the above two outputs.
- You need to implement:
  - the model, using multiple inputs, outputs, losses and metrics;
  - construction of two-image dataset examples using regular MNIST data via the PyTorch datasets.

#### Tests Start: mnist_multiple_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_multiple.py --recodex --epochs=1 --batch_size=50`
```
Epoch 1/1 4.6s loss=0.9001 direct_comparison=0.8716 indirect_comparison=0.9433 dev:loss=0.3282 dev:direct_comparison=0.9500 dev:indirect_comparison=0.9824
```

2. `python3 mnist_multiple.py --recodex --epochs=1 --batch_size=100`
```
Epoch 1/1 4.7s loss=1.1828 direct_comparison=0.8422 indirect_comparison=0.9257 dev:loss=0.4293 dev:direct_comparison=0.9316 dev:indirect_comparison=0.9776
```
#### Tests End:
