### Assignment: mnist_training
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: mnist_training_tests
#### Examples: mnist_training_examples

This exercise should teach you using different optimizers, learning rates,
and learning rate decays. Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl138/tree/master/labs/02/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified learning rate for the optimizer.
- Optionally use a given learning rate schedule. The schedule can be either
  `linear`, `exponential`, or `cosine`. If a schedule is specified, you also
  get a final learning rate, and the learning rate should be gradually decresed
  during training to reach the final learning rate just after the training
  (i.e., the first update after the training would use exactly the final learning rate).

#### Tests Start: mnist_training_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01`
```
accuracy: 0.6537 - loss: 1.2786 - val_accuracy: 0.9098 - val_loss: 0.3743
```

2. `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
accuracy: 0.8221 - loss: 0.6138 - val_accuracy: 0.9492 - val_loss: 0.1873
```

3. `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.1`
```
accuracy: 0.8400 - loss: 0.5742 - val_accuracy: 0.9528 - val_loss: 0.1800
```

4. `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.001`
```
accuracy: 0.8548 - loss: 0.5121 - val_accuracy: 0.9640 - val_loss: 0.1327
```

5. `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.01`
```
accuracy: 0.8858 - loss: 0.3598 - val_accuracy: 0.9564 - val_loss: 0.1393
```

6. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch 1/2 accuracy: 0.8889 - loss: 0.3520 - val_accuracy: 0.9682 - val_loss: 0.1107
Epoch 2/2 accuracy: 0.9715 - loss: 0.0956 - val_accuracy: 0.9792 - val_loss: 0.0688
Next learning rate to be used: 0.0001
```

7. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch 1/2 accuracy: 0.8912 - loss: 0.3447 - val_accuracy: 0.9702 - val_loss: 0.0997
Epoch 2/2 accuracy: 0.9746 - loss: 0.0824 - val_accuracy: 0.9778 - val_loss: 0.0776
Next learning rate to be used: 0.001
```

8. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch 1/2 accuracy: 0.8875 - loss: 0.3548 - val_accuracy: 0.9726 - val_loss: 0.0976
Epoch 2/2 accuracy: 0.9742 - loss: 0.0851 - val_accuracy: 0.9764 - val_loss: 0.0740
Next learning rate to be used: 0.0001
```
#### Tests End:
#### Examples Start: mnist_training_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 accuracy: 0.6537 - loss: 1.2786 - val_accuracy: 0.9098 - val_loss: 0.3743
Epoch  2/10 accuracy: 0.8848 - loss: 0.4316 - val_accuracy: 0.9222 - val_loss: 0.2895
Epoch  3/10 accuracy: 0.9057 - loss: 0.3450 - val_accuracy: 0.9308 - val_loss: 0.2539
Epoch  4/10 accuracy: 0.9118 - loss: 0.3131 - val_accuracy: 0.9372 - val_loss: 0.2368
Epoch  5/10 accuracy: 0.9188 - loss: 0.2924 - val_accuracy: 0.9406 - val_loss: 0.2202
Epoch  6/10 accuracy: 0.9235 - loss: 0.2750 - val_accuracy: 0.9426 - val_loss: 0.2076
Epoch  7/10 accuracy: 0.9291 - loss: 0.2572 - val_accuracy: 0.9464 - val_loss: 0.1997
Epoch  8/10 accuracy: 0.9304 - loss: 0.2456 - val_accuracy: 0.9494 - val_loss: 0.1909
Epoch  9/10 accuracy: 0.9339 - loss: 0.2340 - val_accuracy: 0.9536 - val_loss: 0.1813
Epoch 10/10 accuracy: 0.9377 - loss: 0.2199 - val_accuracy: 0.9534 - val_loss: 0.1756
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 accuracy: 0.8221 - loss: 0.6138 - val_accuracy: 0.9492 - val_loss: 0.1873
Epoch  2/10 accuracy: 0.9370 - loss: 0.2173 - val_accuracy: 0.9646 - val_loss: 0.1385
Epoch  3/10 accuracy: 0.9599 - loss: 0.1453 - val_accuracy: 0.9716 - val_loss: 0.1076
Epoch  4/10 accuracy: 0.9673 - loss: 0.1127 - val_accuracy: 0.9746 - val_loss: 0.0961
Epoch  5/10 accuracy: 0.9740 - loss: 0.0933 - val_accuracy: 0.9774 - val_loss: 0.0875
Epoch  6/10 accuracy: 0.9778 - loss: 0.0811 - val_accuracy: 0.9746 - val_loss: 0.0856
Epoch  7/10 accuracy: 0.9821 - loss: 0.0680 - val_accuracy: 0.9774 - val_loss: 0.0803
Epoch  8/10 accuracy: 0.9825 - loss: 0.0632 - val_accuracy: 0.9776 - val_loss: 0.0780
Epoch  9/10 accuracy: 0.9849 - loss: 0.0552 - val_accuracy: 0.9804 - val_loss: 0.0725
Epoch 10/10 accuracy: 0.9877 - loss: 0.0463 - val_accuracy: 0.9780 - val_loss: 0.0735
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 accuracy: 0.8400 - loss: 0.5742 - val_accuracy: 0.9528 - val_loss: 0.1800
Epoch  2/10 accuracy: 0.9389 - loss: 0.2123 - val_accuracy: 0.9670 - val_loss: 0.1335
Epoch  3/10 accuracy: 0.9602 - loss: 0.1431 - val_accuracy: 0.9728 - val_loss: 0.1052
Epoch  4/10 accuracy: 0.9685 - loss: 0.1115 - val_accuracy: 0.9770 - val_loss: 0.0946
Epoch  5/10 accuracy: 0.9747 - loss: 0.0927 - val_accuracy: 0.9754 - val_loss: 0.0878
Epoch  6/10 accuracy: 0.9775 - loss: 0.0798 - val_accuracy: 0.9754 - val_loss: 0.0852
Epoch  7/10 accuracy: 0.9813 - loss: 0.0680 - val_accuracy: 0.9780 - val_loss: 0.0797
Epoch  8/10 accuracy: 0.9828 - loss: 0.0621 - val_accuracy: 0.9796 - val_loss: 0.0757
Epoch  9/10 accuracy: 0.9847 - loss: 0.0550 - val_accuracy: 0.9804 - val_loss: 0.0731
Epoch 10/10 accuracy: 0.9875 - loss: 0.0464 - val_accuracy: 0.9782 - val_loss: 0.0731
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 accuracy: 0.8548 - loss: 0.5121 - val_accuracy: 0.9640 - val_loss: 0.1327
Epoch  2/10 accuracy: 0.9552 - loss: 0.1505 - val_accuracy: 0.9706 - val_loss: 0.1118
Epoch  3/10 accuracy: 0.9744 - loss: 0.0900 - val_accuracy: 0.9770 - val_loss: 0.0833
Epoch  4/10 accuracy: 0.9808 - loss: 0.0658 - val_accuracy: 0.9778 - val_loss: 0.0786
Epoch  5/10 accuracy: 0.9836 - loss: 0.0533 - val_accuracy: 0.9804 - val_loss: 0.0735
Epoch  6/10 accuracy: 0.9890 - loss: 0.0403 - val_accuracy: 0.9782 - val_loss: 0.0772
Epoch  7/10 accuracy: 0.9911 - loss: 0.0311 - val_accuracy: 0.9792 - val_loss: 0.0756
Epoch  8/10 accuracy: 0.9922 - loss: 0.0257 - val_accuracy: 0.9818 - val_loss: 0.0717
Epoch  9/10 accuracy: 0.9947 - loss: 0.0202 - val_accuracy: 0.9806 - val_loss: 0.0734
Epoch 10/10 accuracy: 0.9953 - loss: 0.0167 - val_accuracy: 0.9802 - val_loss: 0.0779
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 accuracy: 0.8858 - loss: 0.3598 - val_accuracy: 0.9564 - val_loss: 0.1393
Epoch  2/10 accuracy: 0.9565 - loss: 0.1478 - val_accuracy: 0.9622 - val_loss: 0.1445
Epoch  3/10 accuracy: 0.9688 - loss: 0.1041 - val_accuracy: 0.9686 - val_loss: 0.1184
Epoch  4/10 accuracy: 0.9717 - loss: 0.1016 - val_accuracy: 0.9644 - val_loss: 0.1538
Epoch  5/10 accuracy: 0.9749 - loss: 0.0914 - val_accuracy: 0.9642 - val_loss: 0.1477
Epoch  6/10 accuracy: 0.9754 - loss: 0.0878 - val_accuracy: 0.9714 - val_loss: 0.1375
Epoch  7/10 accuracy: 0.9779 - loss: 0.0804 - val_accuracy: 0.9684 - val_loss: 0.1510
Epoch  8/10 accuracy: 0.9793 - loss: 0.0764 - val_accuracy: 0.9696 - val_loss: 0.1803
Epoch  9/10 accuracy: 0.9808 - loss: 0.0747 - val_accuracy: 0.9708 - val_loss: 0.1576
Epoch 10/10 accuracy: 0.9812 - loss: 0.0750 - val_accuracy: 0.9716 - val_loss: 0.1556
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 accuracy: 0.8862 - loss: 0.3582 - val_accuracy: 0.9636 - val_loss: 0.1395
Epoch  2/10 accuracy: 0.9603 - loss: 0.1313 - val_accuracy: 0.9684 - val_loss: 0.1056
Epoch  3/10 accuracy: 0.9730 - loss: 0.0899 - val_accuracy: 0.9718 - val_loss: 0.1089
Epoch  4/10 accuracy: 0.9780 - loss: 0.0701 - val_accuracy: 0.9676 - val_loss: 0.1250
Epoch  5/10 accuracy: 0.9818 - loss: 0.0528 - val_accuracy: 0.9744 - val_loss: 0.1001
Epoch  6/10 accuracy: 0.9876 - loss: 0.0389 - val_accuracy: 0.9738 - val_loss: 0.1233
Epoch  7/10 accuracy: 0.9907 - loss: 0.0255 - val_accuracy: 0.9780 - val_loss: 0.0989
Epoch  8/10 accuracy: 0.9954 - loss: 0.0141 - val_accuracy: 0.9802 - val_loss: 0.0909
Epoch  9/10 accuracy: 0.9976 - loss: 0.0079 - val_accuracy: 0.9814 - val_loss: 0.0923
Epoch 10/10 accuracy: 0.9995 - loss: 0.0033 - val_accuracy: 0.9818 - val_loss: 0.0946
Next learning rate to be used: 0.0001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 accuracy: 0.8877 - loss: 0.3564 - val_accuracy: 0.9616 - val_loss: 0.1278
Epoch  2/10 accuracy: 0.9642 - loss: 0.1228 - val_accuracy: 0.9624 - val_loss: 0.1149
Epoch  3/10 accuracy: 0.9778 - loss: 0.0720 - val_accuracy: 0.9748 - val_loss: 0.0781
Epoch  4/10 accuracy: 0.9844 - loss: 0.0500 - val_accuracy: 0.9750 - val_loss: 0.0973
Epoch  5/10 accuracy: 0.9884 - loss: 0.0356 - val_accuracy: 0.9800 - val_loss: 0.0709
Epoch  6/10 accuracy: 0.9933 - loss: 0.0228 - val_accuracy: 0.9792 - val_loss: 0.0810
Epoch  7/10 accuracy: 0.9956 - loss: 0.0150 - val_accuracy: 0.9806 - val_loss: 0.0785
Epoch  8/10 accuracy: 0.9969 - loss: 0.0095 - val_accuracy: 0.9826 - val_loss: 0.0746
Epoch  9/10 accuracy: 0.9985 - loss: 0.0069 - val_accuracy: 0.9808 - val_loss: 0.0783
Epoch 10/10 accuracy: 0.9994 - loss: 0.0036 - val_accuracy: 0.9818 - val_loss: 0.0783
Next learning rate to be used: 0.001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch  1/10 accuracy: 0.8858 - loss: 0.3601 - val_accuracy: 0.9624 - val_loss: 0.1311
Epoch  2/10 accuracy: 0.9566 - loss: 0.1461 - val_accuracy: 0.9654 - val_loss: 0.1270
Epoch  3/10 accuracy: 0.9695 - loss: 0.1023 - val_accuracy: 0.9740 - val_loss: 0.0965
Epoch  4/10 accuracy: 0.9755 - loss: 0.0790 - val_accuracy: 0.9710 - val_loss: 0.1152
Epoch  5/10 accuracy: 0.9831 - loss: 0.0562 - val_accuracy: 0.9748 - val_loss: 0.1004
Epoch  6/10 accuracy: 0.9889 - loss: 0.0353 - val_accuracy: 0.9758 - val_loss: 0.1003
Epoch  7/10 accuracy: 0.9930 - loss: 0.0206 - val_accuracy: 0.9786 - val_loss: 0.0864
Epoch  8/10 accuracy: 0.9972 - loss: 0.0096 - val_accuracy: 0.9790 - val_loss: 0.0958
Epoch  9/10 accuracy: 0.9985 - loss: 0.0068 - val_accuracy: 0.9802 - val_loss: 0.0880
Epoch 10/10 accuracy: 0.9992 - loss: 0.0042 - val_accuracy: 0.9802 - val_loss: 0.0891
Next learning rate to be used: 0.0001
```
#### Examples End:
