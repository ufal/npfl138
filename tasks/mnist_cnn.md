### Assignment: mnist_cnn
#### Date: Deadline: Mar 26, 22:00
#### Points: 3 points
#### Tests: mnist_cnn_tests
#### Examples: mnist_cnn_examples

To pass this assignment, you will learn to construct basic convolutional
neural network layers. Start with the
[mnist_cnn.py](https://github.com/ufal/npfl138/tree/master/labs/04/mnist_cnn.py)
template and assume the requested architecture is described by the `cnn`
argument, which contains comma-separated specifications of the following layers:
- `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
  activation and specified number of filters, kernel size, stride and padding.
  Example: `C-10-3-1-same`
- `CB-filters-kernel_size-stride-padding`: Same as
  `C-filters-kernel_size-stride-padding`, but use batch normalization.
  In detail, start with a convolutional layer **without bias** and activation,
  then add batch normalization layer, and finally the ReLU activation.
  Example: `CB-10-3-1-same`
- `M-pool_size-stride`: Add max pooling with specified size and stride, using
  the default padding of 0 (the `"valid"` padding).
  Example: `M-3-2`
- `R-[layers]`: Add a residual connection. The `layers` contain a specification
  of at least one convolutional layer (but not a recursive residual connection `R`).
  The input to the `R` layer should be processed sequentially by `layers`, and the
  produced output (after the ReLU nonlinearity of the last layer) should be added
  to the input (of this `R` layer).
  Example: `R-[C-16-3-1-same,C-16-3-1-same]`
- `F`: Flatten inputs. Must appear exactly once in the architecture.
- `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified
  size. Example: `H-100`
- `D-dropout_rate`: Apply dropout with the given dropout rate. Example: `D-0.5`

An example architecture might be `--cnn=CB-16-5-2-valid,M-3-2,F,H-100,D-0.5`.
You can assume the resulting network is valid; it is fine to crash if it is not.

After a successful ReCodEx submission, you can try obtaining the best accuracy
on MNIST and then advance to `cifar_competition`.

#### Tests Start: mnist_cnn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100`
```
Epoch 1/1 1.6s train_loss=0.3178 train_accuracy=0.9104 dev_loss=0.1482 dev_accuracy=0.9566
```

2. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100,D-0.5`
```
Epoch 1/1 1.5s train_loss=0.4831 train_accuracy=0.8574 dev_loss=0.1584 dev_accuracy=0.9556
```

3. `python3 mnist_cnn.py --epochs=1 --cnn=M-5-2,F,H-50`
```
Epoch 1/1 1.4s train_loss=0.7251 train_accuracy=0.7780 dev_loss=0.4007 dev_accuracy=0.8820
```

4. `python3 mnist_cnn.py --epochs=1 --cnn=C-8-3-5-valid,C-8-3-2-valid,F,H-50`
```
Epoch 1/1 1.9s train_loss=0.8031 train_accuracy=0.7437 dev_loss=0.3459 dev_accuracy=0.9000
```

5. `python3 mnist_cnn.py --epochs=1 --cnn=CB-6-3-5-valid,F,H-32`
```
Epoch 1/1 1.7s train_loss=0.6422 train_accuracy=0.8009 dev_loss=0.2784 dev_accuracy=0.9184
```

6. `python3 mnist_cnn.py --epochs=1 --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50`
```
Epoch 1/1 2.6s train_loss=0.4411 train_accuracy=0.8620 dev_loss=0.1888 dev_accuracy=0.9436
```
#### Tests End:
#### Examples Start: mnist_cnn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_cnn.py --cnn=F,H-100`
```
Epoch  1/10 1.4s train_loss=0.3178 train_accuracy=0.9104 dev_loss=0.1482 dev_accuracy=0.9566
Epoch  2/10 1.6s train_loss=0.1502 train_accuracy=0.9560 dev_loss=0.1049 dev_accuracy=0.9720
Epoch  3/10 1.5s train_loss=0.1048 train_accuracy=0.9692 dev_loss=0.0939 dev_accuracy=0.9720
Epoch  4/10 1.5s train_loss=0.0812 train_accuracy=0.9757 dev_loss=0.0856 dev_accuracy=0.9760
Epoch  5/10 1.6s train_loss=0.0627 train_accuracy=0.9811 dev_loss=0.0827 dev_accuracy=0.9786
Epoch  6/10 1.6s train_loss=0.0516 train_accuracy=0.9846 dev_loss=0.0749 dev_accuracy=0.9794
Epoch  7/10 1.6s train_loss=0.0420 train_accuracy=0.9869 dev_loss=0.0726 dev_accuracy=0.9796
Epoch  8/10 1.6s train_loss=0.0331 train_accuracy=0.9903 dev_loss=0.0733 dev_accuracy=0.9800
Epoch  9/10 1.6s train_loss=0.0275 train_accuracy=0.9918 dev_loss=0.0782 dev_accuracy=0.9782
Epoch 10/10 1.6s train_loss=0.0240 train_accuracy=0.9930 dev_loss=0.0782 dev_accuracy=0.9810
```

- `python3 mnist_cnn.py --cnn=F,H-100,D-0.5`
```
Epoch  1/10 1.5s train_loss=0.4831 train_accuracy=0.8574 dev_loss=0.1584 dev_accuracy=0.9556
Epoch  2/10 1.6s train_loss=0.2754 train_accuracy=0.9197 dev_loss=0.1225 dev_accuracy=0.9666
Epoch  3/10 1.7s train_loss=0.2279 train_accuracy=0.9327 dev_loss=0.1021 dev_accuracy=0.9716
Epoch  4/10 1.7s train_loss=0.2027 train_accuracy=0.9401 dev_loss=0.0951 dev_accuracy=0.9714
Epoch  5/10 1.7s train_loss=0.1892 train_accuracy=0.9441 dev_loss=0.0914 dev_accuracy=0.9752
Epoch  6/10 1.7s train_loss=0.1751 train_accuracy=0.9468 dev_loss=0.0821 dev_accuracy=0.9746
Epoch  7/10 1.7s train_loss=0.1659 train_accuracy=0.9495 dev_loss=0.0783 dev_accuracy=0.9760
Epoch  8/10 1.7s train_loss=0.1551 train_accuracy=0.9526 dev_loss=0.0768 dev_accuracy=0.9764
Epoch  9/10 1.7s train_loss=0.1487 train_accuracy=0.9545 dev_loss=0.0828 dev_accuracy=0.9776
Epoch 10/10 1.7s train_loss=0.1446 train_accuracy=0.9548 dev_loss=0.0770 dev_accuracy=0.9776
```

- `python3 mnist_cnn.py --cnn=F,H-200,D-0.5`
```
Epoch  1/10 2.0s train_loss=0.3757 train_accuracy=0.8880 dev_loss=0.1266 dev_accuracy=0.9680
Epoch  2/10 2.2s train_loss=0.1991 train_accuracy=0.9414 dev_loss=0.0981 dev_accuracy=0.9734
Epoch  3/10 2.3s train_loss=0.1595 train_accuracy=0.9528 dev_loss=0.0858 dev_accuracy=0.9762
Epoch  4/10 2.4s train_loss=0.1352 train_accuracy=0.9588 dev_loss=0.0786 dev_accuracy=0.9782
Epoch  5/10 2.4s train_loss=0.1217 train_accuracy=0.9637 dev_loss=0.0739 dev_accuracy=0.9806
Epoch  6/10 2.4s train_loss=0.1089 train_accuracy=0.9669 dev_loss=0.0692 dev_accuracy=0.9818
Epoch  7/10 2.4s train_loss=0.1017 train_accuracy=0.9678 dev_loss=0.0699 dev_accuracy=0.9814
Epoch  8/10 2.4s train_loss=0.0958 train_accuracy=0.9703 dev_loss=0.0691 dev_accuracy=0.9812
Epoch  9/10 2.4s train_loss=0.0859 train_accuracy=0.9723 dev_loss=0.0656 dev_accuracy=0.9828
Epoch 10/10 2.4s train_loss=0.0860 train_accuracy=0.9725 dev_loss=0.0674 dev_accuracy=0.9834
```

- `python3 mnist_cnn.py --cnn=C-8-3-1-same,C-8-3-1-same,M-3-2,C-16-3-1-same,C-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 13.2s train_loss=0.1713 train_accuracy=0.9475 dev_loss=0.0649 dev_accuracy=0.9824
Epoch  2/10 18.8s train_loss=0.0521 train_accuracy=0.9833 dev_loss=0.0369 dev_accuracy=0.9892
Epoch  3/10 19.0s train_loss=0.0383 train_accuracy=0.9881 dev_loss=0.0327 dev_accuracy=0.9916
Epoch  4/10 16.6s train_loss=0.0290 train_accuracy=0.9906 dev_loss=0.0338 dev_accuracy=0.9900
Epoch  5/10 13.1s train_loss=0.0247 train_accuracy=0.9922 dev_loss=0.0330 dev_accuracy=0.9898
Epoch  6/10 13.1s train_loss=0.0201 train_accuracy=0.9935 dev_loss=0.0369 dev_accuracy=0.9902
Epoch  7/10 16.1s train_loss=0.0176 train_accuracy=0.9945 dev_loss=0.0358 dev_accuracy=0.9910
Epoch  8/10 19.0s train_loss=0.0147 train_accuracy=0.9953 dev_loss=0.0315 dev_accuracy=0.9932
Epoch  9/10 18.4s train_loss=0.0138 train_accuracy=0.9954 dev_loss=0.0278 dev_accuracy=0.9932
Epoch 10/10 13.2s train_loss=0.0104 train_accuracy=0.9966 dev_loss=0.0334 dev_accuracy=0.9922
```

- `python3 mnist_cnn.py --cnn=CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 14.3s train_loss=0.1443 train_accuracy=0.9569 dev_loss=0.0501 dev_accuracy=0.9854
Epoch  2/10 19.5s train_loss=0.0517 train_accuracy=0.9838 dev_loss=0.0678 dev_accuracy=0.9792
Epoch  3/10 20.7s train_loss=0.0399 train_accuracy=0.9873 dev_loss=0.0369 dev_accuracy=0.9892
Epoch  4/10 20.5s train_loss=0.0337 train_accuracy=0.9893 dev_loss=0.0376 dev_accuracy=0.9900
Epoch  5/10 17.7s train_loss=0.0281 train_accuracy=0.9908 dev_loss=0.0264 dev_accuracy=0.9926
Epoch  6/10 14.9s train_loss=0.0226 train_accuracy=0.9929 dev_loss=0.0384 dev_accuracy=0.9900
Epoch  7/10 14.9s train_loss=0.0203 train_accuracy=0.9935 dev_loss=0.0516 dev_accuracy=0.9864
Epoch  8/10 21.7s train_loss=0.0179 train_accuracy=0.9941 dev_loss=0.0381 dev_accuracy=0.9908
Epoch  9/10 20.4s train_loss=0.0151 train_accuracy=0.9949 dev_loss=0.0348 dev_accuracy=0.9918
Epoch 10/10 19.0s train_loss=0.0157 train_accuracy=0.9946 dev_loss=0.0319 dev_accuracy=0.9918
```
#### Examples End:
