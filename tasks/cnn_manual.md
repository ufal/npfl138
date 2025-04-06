### Assignment: cnn_manual
#### Date: Deadline: Apr 09, 22:00
#### Points: 3 points
#### Slides@: https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/slides/?cnn_manual
#### Video@: https://lectures.ms.mff.cuni.cz/video/rec/npfl138/2425/npfl138-2425-06-english.cnn_manual.mp4, Recording
#### Tests: cnn_manual_tests

To pass this assignment, you need to manually implement the forward and backward
pass through a 2D convolutional layer. Start with the
[cnn_manual.py](https://github.com/ufal/npfl138/tree/master/labs/05/cnn_manual.py)
template, which constructs a series of 2D convolutional layers with ReLU
activation and `valid` padding, specified in the `args.cnn` option.
The `args.cnn` contains comma-separated layer specifications in the format
`filters-kernel_size-stride`.

In this assignment, we use the `channels_last` (`NHWC`) format; therefore,
images have shape `[batch_size, height, width, channels]` and the convolutional
kernel has shape `[kernel_height, kernel_width, in_channels, out_channels]`.
These shapes are consistent with the course slides, but are different from the
native PyTorch format.

Of course, you cannot use any PyTorch convolutional operation (including
`torch.nn.{Fold/Unfold}` and `torch.nn.functional.{fold/unfold}`) nor
the `.backward()` for gradient computation. Instead, implement convolution
and gradient computations using matrix multiplication and other basic
operations (element-wise multiplication, summation, etc; if you want, you can
read about `torch.einsum`).

To make debugging easier, the template supports a `--verify` option, which
allows comparing the forward pass and the three gradients you compute in the
backward pass to correct values.

#### Tests Start: cnn_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 cnn_manual.py --epochs=1 --cnn=5-1-1`
```
Dev accuracy after epoch 1 is 88.16
Test accuracy after epoch 1 is 85.91
```

2. `python3 cnn_manual.py --epochs=1 --cnn=5-3-1`
```
Dev accuracy after epoch 1 is 88.26
Test accuracy after epoch 1 is 86.12
```

3. `python3 cnn_manual.py --epochs=1 --cnn=5-3-2`
```
Dev accuracy after epoch 1 is 87.44
Test accuracy after epoch 1 is 84.97
```

4. `python3 cnn_manual.py --epochs=1 --cnn=5-3-2,10-3-2`
```
Dev accuracy after epoch 1 is 85.14
Test accuracy after epoch 1 is 82.26
```

5. `python3 cnn_manual.py --epochs=1 --cnn=30-1-1,20-3-2`
```
Dev accuracy after epoch 1 is 89.36
Test accuracy after epoch 1 is 86.77
```
#### Tests End:
