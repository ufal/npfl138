### Assignment: mnist_multiple
#### Date: Deadline: Mar 26, 22:00
#### Points: 3 points
#### Tests: mnist_multiple_tests

In this assignment you will implement a model with multiple inputs and outputs.
Start with the [mnist_multiple.py](https://github.com/ufal/npfl138/tree/master/labs/04/mnist_multiple.py)
template and:

![mnist_multiple](//ufal.mff.cuni.cz/~straka/courses/npfl138/2324/tasks/figures/mnist_multiple.svgz)

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
direct_comparison_accuracy: 0.7993 - indirect_comparison_accuracy: 0.8930 - loss: 1.6710 - val_direct_comparison_accuracy: 0.9508 - val_indirect_comparison_accuracy: 0.9836 - val_loss: 0.2984
```

2. `python3 mnist_multiple.py --epochs=1 --batch_size=100`
```
direct_comparison_accuracy: 0.7680 - indirect_comparison_accuracy: 0.8637 - loss: 2.1429 - val_direct_comparison_accuracy: 0.9288 - val_indirect_comparison_accuracy: 0.9772 - val_loss: 0.4157
```
#### Tests End:
