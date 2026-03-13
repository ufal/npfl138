### Assignment: mnist_cnn
#### Date: Deadline: Mar 25, 22:00
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

1. `python3 mnist_cnn.py --recodex --epochs=1 --cnn=F,H-100`
```
Epoch 1/1 1.9s loss=0.3149 accuracy=0.9109 dev:loss=0.1453 dev:accuracy=0.9606
```

2. `python3 mnist_cnn.py --recodex --epochs=1 --cnn=F,H-100,D-0.5`
```
Epoch 1/1 2.1s loss=0.4747 accuracy=0.8600 dev:loss=0.1621 dev:accuracy=0.9548
```

3. `python3 mnist_cnn.py --recodex --epochs=1 --cnn=M-5-2,F,H-50`
```
Epoch 1/1 1.3s loss=0.7224 accuracy=0.7815 dev:loss=0.4016 dev:accuracy=0.8794
```

4. `python3 mnist_cnn.py --recodex --epochs=1 --cnn=C-8-3-5-valid,C-8-3-2-valid,F,H-50`
```
Epoch 1/1 1.6s loss=0.8381 accuracy=0.7303 dev:loss=0.3949 dev:accuracy=0.8798
```

5. `python3 mnist_cnn.py --recodex --epochs=1 --cnn=CB-6-3-5-valid,F,H-32`
```
Epoch 1/1 1.6s loss=0.5869 accuracy=0.8192 dev:loss=0.2439 dev:accuracy=0.9262
```

6. `python3 mnist_cnn.py --recodex --epochs=1 --cnn="CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50"`
```
Epoch 1/1 3.2s loss=0.4437 accuracy=0.8620 dev:loss=0.1763 dev:accuracy=0.9480
```
#### Tests End:
#### Examples Start: mnist_cnn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_cnn.py --cnn=F,H-100`
```
Epoch  1/10 1.4s loss=0.3139 accuracy=0.9125 dev:loss=0.1444 dev:accuracy=0.9600
Epoch  2/10 1.5s loss=0.1428 accuracy=0.9587 dev:loss=0.0997 dev:accuracy=0.9736
Epoch  3/10 1.4s loss=0.1005 accuracy=0.9699 dev:loss=0.0912 dev:accuracy=0.9736
Epoch  4/10 1.4s loss=0.0784 accuracy=0.9761 dev:loss=0.0814 dev:accuracy=0.9768
Epoch  5/10 1.4s loss=0.0605 accuracy=0.9816 dev:loss=0.0822 dev:accuracy=0.9776
Epoch  6/10 1.4s loss=0.0498 accuracy=0.9850 dev:loss=0.0747 dev:accuracy=0.9802
Epoch  7/10 1.4s loss=0.0409 accuracy=0.9878 dev:loss=0.0712 dev:accuracy=0.9802
Epoch  8/10 1.4s loss=0.0331 accuracy=0.9900 dev:loss=0.0797 dev:accuracy=0.9786
Epoch  9/10 1.4s loss=0.0272 accuracy=0.9924 dev:loss=0.0764 dev:accuracy=0.9784
Epoch 10/10 1.4s loss=0.0231 accuracy=0.9937 dev:loss=0.0835 dev:accuracy=0.9804
```

- `python3 mnist_cnn.py --cnn=F,H-100,D-0.5`
```
Epoch  1/10 1.4s loss=0.4925 accuracy=0.8536 dev:loss=0.1637 dev:accuracy=0.9576
Epoch  2/10 1.5s loss=0.2744 accuracy=0.9194 dev:loss=0.1220 dev:accuracy=0.9672
Epoch  3/10 1.5s loss=0.2329 accuracy=0.9305 dev:loss=0.1079 dev:accuracy=0.9714
Epoch  4/10 1.5s loss=0.2060 accuracy=0.9389 dev:loss=0.0983 dev:accuracy=0.9720
Epoch  5/10 1.6s loss=0.1912 accuracy=0.9422 dev:loss=0.0899 dev:accuracy=0.9764
Epoch  6/10 1.8s loss=0.1783 accuracy=0.9461 dev:loss=0.0874 dev:accuracy=0.9756
Epoch  7/10 1.7s loss=0.1698 accuracy=0.9485 dev:loss=0.0911 dev:accuracy=0.9762
Epoch  8/10 1.7s loss=0.1597 accuracy=0.9509 dev:loss=0.0822 dev:accuracy=0.9770
Epoch  9/10 1.6s loss=0.1492 accuracy=0.9536 dev:loss=0.0839 dev:accuracy=0.9782
Epoch 10/10 1.7s loss=0.1472 accuracy=0.9540 dev:loss=0.0834 dev:accuracy=0.9768
```

- `python3 mnist_cnn.py --cnn=F,H-200,D-0.5`
```
Epoch  1/10 1.9s loss=0.3752 accuracy=0.8890 dev:loss=0.1360 dev:accuracy=0.9636
Epoch  2/10 2.1s loss=0.1984 accuracy=0.9419 dev:loss=0.1022 dev:accuracy=0.9714
Epoch  3/10 2.1s loss=0.1575 accuracy=0.9530 dev:loss=0.0873 dev:accuracy=0.9770
Epoch  4/10 2.2s loss=0.1382 accuracy=0.9579 dev:loss=0.0771 dev:accuracy=0.9776
Epoch  5/10 2.2s loss=0.1192 accuracy=0.9630 dev:loss=0.0734 dev:accuracy=0.9798
Epoch  6/10 2.2s loss=0.1113 accuracy=0.9655 dev:loss=0.0687 dev:accuracy=0.9802
Epoch  7/10 2.2s loss=0.1000 accuracy=0.9692 dev:loss=0.0697 dev:accuracy=0.9808
Epoch  8/10 2.3s loss=0.0951 accuracy=0.9701 dev:loss=0.0657 dev:accuracy=0.9826
Epoch  9/10 2.2s loss=0.0888 accuracy=0.9719 dev:loss=0.0658 dev:accuracy=0.9822
Epoch 10/10 2.2s loss=0.0804 accuracy=0.9738 dev:loss=0.0688 dev:accuracy=0.9822
```

- `python3 mnist_cnn.py --cnn=C-8-3-1-same,C-8-3-1-same,M-3-2,C-16-3-1-same,C-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 12.4s loss=0.1789 accuracy=0.9453 dev:loss=0.0655 dev:accuracy=0.9806
Epoch  2/10 16.1s loss=0.0535 accuracy=0.9837 dev:loss=0.0414 dev:accuracy=0.9870
Epoch  3/10 16.8s loss=0.0400 accuracy=0.9872 dev:loss=0.0341 dev:accuracy=0.9908
Epoch  4/10 15.3s loss=0.0323 accuracy=0.9899 dev:loss=0.0372 dev:accuracy=0.9892
Epoch  5/10 11.7s loss=0.0250 accuracy=0.9919 dev:loss=0.0274 dev:accuracy=0.9924
Epoch  6/10 11.8s loss=0.0224 accuracy=0.9925 dev:loss=0.0454 dev:accuracy=0.9860
Epoch  7/10 13.8s loss=0.0186 accuracy=0.9940 dev:loss=0.0414 dev:accuracy=0.9886
Epoch  8/10 16.7s loss=0.0159 accuracy=0.9947 dev:loss=0.0310 dev:accuracy=0.9918
Epoch  9/10 16.7s loss=0.0147 accuracy=0.9951 dev:loss=0.0383 dev:accuracy=0.9890
Epoch 10/10 15.8s loss=0.0135 accuracy=0.9956 dev:loss=0.0336 dev:accuracy=0.9906
```

- `python3 mnist_cnn.py --cnn=CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 13.3s loss=0.1430 accuracy=0.9547 dev:loss=0.0592 dev:accuracy=0.9816
Epoch  2/10 14.0s loss=0.0512 accuracy=0.9841 dev:loss=0.0593 dev:accuracy=0.9818
Epoch  3/10 18.9s loss=0.0390 accuracy=0.9880 dev:loss=0.0430 dev:accuracy=0.9872
Epoch  4/10 19.4s loss=0.0339 accuracy=0.9891 dev:loss=0.0336 dev:accuracy=0.9904
Epoch  5/10 18.7s loss=0.0276 accuracy=0.9912 dev:loss=0.0373 dev:accuracy=0.9872
Epoch  6/10 13.5s loss=0.0228 accuracy=0.9926 dev:loss=0.0435 dev:accuracy=0.9868
Epoch  7/10 13.9s loss=0.0216 accuracy=0.9932 dev:loss=0.0502 dev:accuracy=0.9850
Epoch  8/10 19.7s loss=0.0184 accuracy=0.9937 dev:loss=0.0556 dev:accuracy=0.9860
Epoch  9/10 19.3s loss=0.0154 accuracy=0.9948 dev:loss=0.0252 dev:accuracy=0.9936
Epoch 10/10 18.8s loss=0.0138 accuracy=0.9952 dev:loss=0.0321 dev:accuracy=0.9912
```
#### Examples End:
