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
  In detail, start with a convolutional layer **without bias and activation**,
  then add batch normalization layer, and finally ReLU activation.
  Example: `CB-10-3-1-same`
- `M-pool_size-stride`: Add max pooling with specified size and stride, using
  the default `"valid"` padding.
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

An example architecture might be `--cnn=CB-16-5-2-same,M-3-2,F,H-100,D-0.5`.
You can assume the resulting network is valid; it is fine to crash if it is not.

After a successful ReCodEx submission, you can try obtaining the best accuracy
on MNIST and then advance to `cifar_competition`.

#### Tests Start: mnist_cnn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100`
```
accuracy: 0.8503 - loss: 0.5286 - val_accuracy: 0.9604 - val_loss: 0.1432
```

2. `python3 mnist_cnn.py --epochs=1 --cnn=F,H-100,D-0.5`
```
accuracy: 0.7706 - loss: 0.7444 - val_accuracy: 0.9572 - val_loss: 0.1606
```

3. `python3 mnist_cnn.py --epochs=1 --cnn=M-5-2,F,H-50`
```
accuracy: 0.6630 - loss: 1.0703 - val_accuracy: 0.8798 - val_loss: 0.3894
```

4. `python3 mnist_cnn.py --epochs=1 --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50`
```
accuracy: 0.5898 - loss: 1.2535 - val_accuracy: 0.8774 - val_loss: 0.4079
```

5. `python3 mnist_cnn.py --epochs=1 --cnn=CB-6-3-5-valid,F,H-32`
```
accuracy: 0.6822 - loss: 1.0011 - val_accuracy: 0.9284 - val_loss: 0.2537
```

6. `python3 mnist_cnn.py --epochs=1 --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50`
```
accuracy: 0.7562 - loss: 0.7717 - val_accuracy: 0.9486 - val_loss: 0.1734
```
#### Tests End:
#### Examples Start: mnist_cnn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_cnn.py --cnn=F,H-100`
```
Epoch  1/10 accuracy: 0.8503 - loss: 0.5286 - val_accuracy: 0.9604 - val_loss: 0.1432
Epoch  2/10 accuracy: 0.9508 - loss: 0.1654 - val_accuracy: 0.9650 - val_loss: 0.1245
Epoch  3/10 accuracy: 0.9710 - loss: 0.1034 - val_accuracy: 0.9738 - val_loss: 0.0916
Epoch  4/10 accuracy: 0.9773 - loss: 0.0774 - val_accuracy: 0.9762 - val_loss: 0.0848
Epoch  5/10 accuracy: 0.9824 - loss: 0.0613 - val_accuracy: 0.9808 - val_loss: 0.0740
Epoch  6/10 accuracy: 0.9857 - loss: 0.0485 - val_accuracy: 0.9760 - val_loss: 0.0761
Epoch  7/10 accuracy: 0.9893 - loss: 0.0373 - val_accuracy: 0.9770 - val_loss: 0.0774
Epoch  8/10 accuracy: 0.9911 - loss: 0.0323 - val_accuracy: 0.9774 - val_loss: 0.0813
Epoch  9/10 accuracy: 0.9922 - loss: 0.0271 - val_accuracy: 0.9794 - val_loss: 0.0819
Epoch 10/10 accuracy: 0.9948 - loss: 0.0202 - val_accuracy: 0.9788 - val_loss: 0.0821
```

- `python3 mnist_cnn.py --cnn=F,H-100,D-0.5`
```
Epoch  1/10 accuracy: 0.7706 - loss: 0.7444 - val_accuracy: 0.9572 - val_loss: 0.1606
Epoch  2/10 accuracy: 0.9177 - loss: 0.2808 - val_accuracy: 0.9646 - val_loss: 0.1286
Epoch  3/10 accuracy: 0.9313 - loss: 0.2340 - val_accuracy: 0.9732 - val_loss: 0.1038
Epoch  4/10 accuracy: 0.9389 - loss: 0.2025 - val_accuracy: 0.9730 - val_loss: 0.0951
Epoch  5/10 accuracy: 0.9409 - loss: 0.1919 - val_accuracy: 0.9752 - val_loss: 0.0927
Epoch  6/10 accuracy: 0.9448 - loss: 0.1784 - val_accuracy: 0.9768 - val_loss: 0.0864
Epoch  7/10 accuracy: 0.9495 - loss: 0.1649 - val_accuracy: 0.9758 - val_loss: 0.0833
Epoch  8/10 accuracy: 0.9506 - loss: 0.1577 - val_accuracy: 0.9768 - val_loss: 0.0826
Epoch  9/10 accuracy: 0.9544 - loss: 0.1496 - val_accuracy: 0.9778 - val_loss: 0.0806
Epoch 10/10 accuracy: 0.9560 - loss: 0.1413 - val_accuracy: 0.9754 - val_loss: 0.0792
```

- `python3 mnist_cnn.py --cnn=F,H-200,D-0.5`
```
Epoch  1/10 accuracy: 0.8109 - loss: 0.6191 - val_accuracy: 0.9654 - val_loss: 0.1286
Epoch  2/10 accuracy: 0.9382 - loss: 0.2101 - val_accuracy: 0.9718 - val_loss: 0.0995
Epoch  3/10 accuracy: 0.9530 - loss: 0.1598 - val_accuracy: 0.9752 - val_loss: 0.0820
Epoch  4/10 accuracy: 0.9586 - loss: 0.1377 - val_accuracy: 0.9792 - val_loss: 0.0758
Epoch  5/10 accuracy: 0.9635 - loss: 0.1233 - val_accuracy: 0.9792 - val_loss: 0.0684
Epoch  6/10 accuracy: 0.9639 - loss: 0.1133 - val_accuracy: 0.9800 - val_loss: 0.0709
Epoch  7/10 accuracy: 0.9698 - loss: 0.1003 - val_accuracy: 0.9822 - val_loss: 0.0647
Epoch  8/10 accuracy: 0.9701 - loss: 0.0945 - val_accuracy: 0.9814 - val_loss: 0.0626
Epoch  9/10 accuracy: 0.9720 - loss: 0.0886 - val_accuracy: 0.9810 - val_loss: 0.0658
Epoch 10/10 accuracy: 0.9727 - loss: 0.0843 - val_accuracy: 0.9816 - val_loss: 0.0643
```

- `python3 mnist_cnn.py --cnn=C-8-3-1-same,C-8-3-1-same,M-3-2,C-16-3-1-same,C-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 accuracy: 0.8549 - loss: 0.4564 - val_accuracy: 0.9836 - val_loss: 0.0529
Epoch  2/10 accuracy: 0.9809 - loss: 0.0610 - val_accuracy: 0.9830 - val_loss: 0.0527
Epoch  3/10 accuracy: 0.9878 - loss: 0.0406 - val_accuracy: 0.9902 - val_loss: 0.0303
Epoch  4/10 accuracy: 0.9905 - loss: 0.0309 - val_accuracy: 0.9872 - val_loss: 0.0444
Epoch  5/10 accuracy: 0.9916 - loss: 0.0247 - val_accuracy: 0.9918 - val_loss: 0.0286
Epoch  6/10 accuracy: 0.9930 - loss: 0.0214 - val_accuracy: 0.9924 - val_loss: 0.0286
Epoch  7/10 accuracy: 0.9941 - loss: 0.0184 - val_accuracy: 0.9910 - val_loss: 0.0318
Epoch  8/10 accuracy: 0.9955 - loss: 0.0135 - val_accuracy: 0.9944 - val_loss: 0.0236
Epoch  9/10 accuracy: 0.9963 - loss: 0.0116 - val_accuracy: 0.9928 - val_loss: 0.0262
Epoch 10/10 accuracy: 0.9953 - loss: 0.0126 - val_accuracy: 0.9916 - val_loss: 0.0309
```

- `python3 mnist_cnn.py --cnn=CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-200`
```
Epoch  1/10 accuracy: 0.8951 - loss: 0.3258 - val_accuracy: 0.9868 - val_loss: 0.0435
Epoch  2/10 accuracy: 0.9834 - loss: 0.0514 - val_accuracy: 0.9866 - val_loss: 0.0479
Epoch  3/10 accuracy: 0.9879 - loss: 0.0401 - val_accuracy: 0.9898 - val_loss: 0.0351
Epoch  4/10 accuracy: 0.9904 - loss: 0.0297 - val_accuracy: 0.9886 - val_loss: 0.0441
Epoch  5/10 accuracy: 0.9918 - loss: 0.0245 - val_accuracy: 0.9940 - val_loss: 0.0233
Epoch  6/10 accuracy: 0.9937 - loss: 0.0195 - val_accuracy: 0.9898 - val_loss: 0.0336
Epoch  7/10 accuracy: 0.9934 - loss: 0.0203 - val_accuracy: 0.9934 - val_loss: 0.0229
Epoch  8/10 accuracy: 0.9951 - loss: 0.0139 - val_accuracy: 0.9938 - val_loss: 0.0260
Epoch  9/10 accuracy: 0.9958 - loss: 0.0127 - val_accuracy: 0.9938 - val_loss: 0.0248
Epoch 10/10 accuracy: 0.9954 - loss: 0.0132 - val_accuracy: 0.9934 - val_loss: 0.0217
```
#### Examples End:
