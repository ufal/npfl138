### Assignment: pca_first
#### Date: Deadline: Mar 05, 21:59 a.m.
#### Points: 2 points
#### Tests: pca_first_tests

The goal of this exercise is to familiarize with PyTorch `torch.Tensor`s,
shapes and basic tensor manipulation methods. Start with the
[pca_first.py](https://github.com/ufal/npfl138/tree/master/labs/01/pca_first.py)
(and you will also need the [mnist.py](https://github.com/ufal/npfl138/tree/master/labs/01/mnist.py)
module).

Alternatively, you can instead use the
[pca_first.keras.py](https://github.com/ufal/npfl138/tree/master/labs/01/pca_first.keras.py)
template, which uses backend-agnostic `keras.ops` operations instead of PyTorch
operations – both templates can be used to solve the assignment.

In this assignment, you should compute the covariance matrix of several examples
from the MNIST dataset, then compute the first principal component, and quantify
the explained variance of it. It is fine if you are not familiar with terms like
covariance matrix or principal component – the template contains a detailed
description of what you have to do.

Finally, you might want to read the [Introduction to PyTorch
Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html).

#### Tests Start: pca_first_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 pca_first.py --examples=1024 --iterations=64`
```
Total variance: 53.12
Explained variance: 9.64%
```

2. `python3 pca_first.py --examples=8192 --iterations=128`
```
Total variance: 53.05
Explained variance: 9.89%
```

3. `python3 pca_first.py --examples=55000 --iterations=1024`
```
Total variance: 52.74
Explained variance: 9.71%
```
#### Tests End:
