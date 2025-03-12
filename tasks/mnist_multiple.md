### Assignment: mnist_multiple
#### Date: Deadline: Mar 26, 22:00
#### Points: 3 points
#### Tests: mnist_multiple_tests

In this assignment you will implement a model with multiple inputs and outputs.
Start with the [mnist_multiple.py](https://github.com/ufal/npfl138/tree/master/labs/04/mnist_multiple.py)
template and:

![mnist_multiple](//ufal.mff.cuni.cz/~straka/courses/npfl138/2425/tasks/figures/mnist_multiple.svgz)

- The goal is to create a model, which given two input MNIST images, compares if the
  digit on the first one is greater than on the second one.
- We perform this comparison in two different ways:
  - first by directly predicting the comparison by the network (_direct comparison_),
  - then by first classifying the images into digits and then comparing these predictions (_indirect comparison_).
- The model has four outputs:
  - _direct comparison_ whether the first digit is greater than the second one,
  - digit classification for the first image,
  - digit classification for the second image,
  - _indirect comparison_ comparing the digits predicted by the above two outputs.
- You need to implement:
  - the model, using multiple inputs, outputs, losses and metrics;
  - construction of two-image dataset examples using regular MNIST data via the PyTorch datasets.

#### Tests Start: mnist_multiple_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_multiple.py --epochs=1 --batch_size=50`
```
Epoch 1/1 3.8s train_loss=0.8910 train_direct_comparison=0.8716 train_indirect_comparison=0.9439 dev_loss=0.2992 dev_direct_comparison=0.9544 dev_indirect_comparison=0.9832
```

2. `python3 mnist_multiple.py --epochs=1 --batch_size=100`
```
Epoch 1/1 3.6s train_loss=1.1647 train_direct_comparison=0.8425 train_indirect_comparison=0.9268 dev_loss=0.4307 dev_direct_comparison=0.9284 dev_indirect_comparison=0.9772
```
#### Tests End:
