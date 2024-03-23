### Assignment: cnn_manual
#### Date: Deadline: Apr 02, 22:00
#### Points: 3 points
#### Slides@: https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/slides/?cnn_manual
#### Tests: cnn_manual_tests

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
[cnn_manual.py](https://github.com/ufal/npfl138/tree/master/labs/05/cnn_manual.py)
template, which constructs a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma-separated layer specifications in the format
`filters-kernel_size-stride`.

Of course, you cannot use any PyTorch convolutional operation (instead,
implement the forward and backward pass using matrix multiplication and other
operations), nor the `.backward()` for gradient computation.

To make debugging easier, the template supports a `--verify` option, which
allows comparing the forward pass and the three gradients you compute in the
backward pass to correct values.

#### Tests Start: cnn_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 cnn_manual.py --epochs=1 --cnn=5-1-1`
```
Dev accuracy after epoch 1 is 90.42
Test accuracy after epoch 1 is 88.55
```

2. `python3 cnn_manual.py --epochs=1 --cnn=5-3-1`
```
Dev accuracy after epoch 1 is 92.26
Test accuracy after epoch 1 is 90.59
```

3. `python3 cnn_manual.py --epochs=1 --cnn=5-3-2`
```
Dev accuracy after epoch 1 is 90.82
Test accuracy after epoch 1 is 88.78
```

4. `python3 cnn_manual.py --epochs=1 --cnn=5-3-2,10-3-2`
```
Dev accuracy after epoch 1 is 92.92
Test accuracy after epoch 1 is 90.97
```
#### Tests End:
