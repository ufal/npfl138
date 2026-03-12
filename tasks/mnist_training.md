### Assignment: mnist_training
#### Date: Deadline: Mar 11, 22:00
#### Points: 2 points
#### Tests: mnist_training_tests
#### Examples: mnist_training_examples

This exercise should teach you to use different optimizers, learning rates,
and learning rate decays from [torch.optim](https://docs.pytorch.org/docs/stable/optim.html). Your goal is to modify the
[mnist_training.py](https://github.com/ufal/npfl138/tree/master/labs/02/mnist_training.py)
template and implement the following:
- Using specified optimizer (either `SGD` or `Adam`).
- Optionally using momentum for the `SGD` optimizer.
- Using specified learning rate for the optimizer.
- Optionally use a given learning rate schedule. The schedule can be either
  `linear`, `exponential`, or `cosine`. If a schedule is specified, you also
  get a final learning rate, and you should gradually decrease the learning rate
  during the training to reach the final learning rate just after the training
  (i.e., the first update after the training would use exactly the final learning rate).

#### Tests Start: mnist_training_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_training.py --recodex --epochs=1 --optimizer=SGD --learning_rate=0.01`
```
Epoch 1/1 1.4s loss=0.8234 accuracy=0.7983 dev:loss=0.3712 dev:accuracy=0.9104
```

2. `python3 mnist_training.py --recodex --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch 1/1 1.5s loss=0.3665 accuracy=0.8955 dev:loss=0.1809 dev:accuracy=0.9524
```

3. `python3 mnist_training.py --recodex --epochs=1 --optimizer=SGD --learning_rate=0.1`
```
Epoch 1/1 1.4s loss=0.3580 accuracy=0.8987 dev:loss=0.1707 dev:accuracy=0.9542
```

4. `python3 mnist_training.py --recodex --epochs=1 --optimizer=Adam --learning_rate=0.001`
```
Epoch 1/1 1.9s loss=0.2982 accuracy=0.9153 dev:loss=0.1324 dev:accuracy=0.9640
```

5. `python3 mnist_training.py --recodex --epochs=1 --optimizer=Adam --learning_rate=0.01`
```
Epoch 1/1 2.0s loss=0.2293 accuracy=0.9298 dev:loss=0.1437 dev:accuracy=0.9598
```

6. `python3 mnist_training.py --recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch 1/2 2.1s lr=0.0050 loss=0.2097 accuracy=0.9365 dev:loss=0.1113 dev:accuracy=0.9688
Epoch 2/2 2.3s lr=1.00e-04 loss=0.0752 accuracy=0.9772 dev:loss=0.0752 dev:accuracy=0.9800
Next learning rate to be used: 0.0001
```

7. `python3 mnist_training.py --recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch 1/2 2.0s lr=0.0032 loss=0.2013 accuracy=0.9392 dev:loss=0.1019 dev:accuracy=0.9694
Epoch 2/2 2.3s lr=0.0010 loss=0.0734 accuracy=0.9778 dev:loss=0.0737 dev:accuracy=0.9800
Next learning rate to be used: 0.001
```

8. `python3 mnist_training.py --recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch 1/2 2.1s lr=0.0050 loss=0.2170 accuracy=0.9337 dev:loss=0.1177 dev:accuracy=0.9662
Epoch 2/2 2.4s lr=1.00e-04 loss=0.0731 accuracy=0.9772 dev:loss=0.0788 dev:accuracy=0.9786
Next learning rate to be used: 0.0001
```
#### Tests End:
#### Examples Start: mnist_training_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 1.4s loss=0.8300 accuracy=0.7960 dev:loss=0.3780 dev:accuracy=0.9060
Epoch  2/10 1.4s loss=0.4088 accuracy=0.8892 dev:loss=0.2940 dev:accuracy=0.9208
Epoch  3/10 1.4s loss=0.3473 accuracy=0.9030 dev:loss=0.2585 dev:accuracy=0.9286
Epoch  4/10 1.4s loss=0.3144 accuracy=0.9116 dev:loss=0.2383 dev:accuracy=0.9352
Epoch  5/10 1.4s loss=0.2911 accuracy=0.9184 dev:loss=0.2230 dev:accuracy=0.9404
Epoch  6/10 1.4s loss=0.2729 accuracy=0.9235 dev:loss=0.2093 dev:accuracy=0.9432
Epoch  7/10 1.4s loss=0.2577 accuracy=0.9281 dev:loss=0.1993 dev:accuracy=0.9480
Epoch  8/10 1.4s loss=0.2442 accuracy=0.9316 dev:loss=0.1903 dev:accuracy=0.9510
Epoch  9/10 1.4s loss=0.2326 accuracy=0.9350 dev:loss=0.1828 dev:accuracy=0.9546
Epoch 10/10 1.4s loss=0.2222 accuracy=0.9379 dev:loss=0.1744 dev:accuracy=0.9546
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 1.5s loss=0.3731 accuracy=0.8952 dev:loss=0.1912 dev:accuracy=0.9472
Epoch  2/10 1.7s loss=0.1942 accuracy=0.9437 dev:loss=0.1322 dev:accuracy=0.9662
Epoch  3/10 1.7s loss=0.1432 accuracy=0.9588 dev:loss=0.1137 dev:accuracy=0.9688
Epoch  4/10 1.7s loss=0.1148 accuracy=0.9674 dev:loss=0.0954 dev:accuracy=0.9744
Epoch  5/10 1.7s loss=0.0962 accuracy=0.9728 dev:loss=0.0914 dev:accuracy=0.9740
Epoch  6/10 1.7s loss=0.0824 accuracy=0.9767 dev:loss=0.0823 dev:accuracy=0.9772
Epoch  7/10 1.7s loss=0.0718 accuracy=0.9801 dev:loss=0.0806 dev:accuracy=0.9780
Epoch  8/10 1.8s loss=0.0640 accuracy=0.9817 dev:loss=0.0741 dev:accuracy=0.9800
Epoch  9/10 1.7s loss=0.0565 accuracy=0.9841 dev:loss=0.0775 dev:accuracy=0.9800
Epoch 10/10 1.8s loss=0.0509 accuracy=0.9861 dev:loss=0.0737 dev:accuracy=0.9788
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 1.4s loss=0.3660 accuracy=0.8970 dev:loss=0.1945 dev:accuracy=0.9460
Epoch  2/10 1.4s loss=0.1940 accuracy=0.9438 dev:loss=0.1320 dev:accuracy=0.9652
Epoch  3/10 1.4s loss=0.1433 accuracy=0.9588 dev:loss=0.1101 dev:accuracy=0.9696
Epoch  4/10 1.4s loss=0.1146 accuracy=0.9673 dev:loss=0.0941 dev:accuracy=0.9748
Epoch  5/10 1.4s loss=0.0949 accuracy=0.9735 dev:loss=0.0915 dev:accuracy=0.9754
Epoch  6/10 1.4s loss=0.0816 accuracy=0.9766 dev:loss=0.0804 dev:accuracy=0.9782
Epoch  7/10 1.4s loss=0.0714 accuracy=0.9800 dev:loss=0.0783 dev:accuracy=0.9792
Epoch  8/10 1.4s loss=0.0627 accuracy=0.9819 dev:loss=0.0734 dev:accuracy=0.9804
Epoch  9/10 1.4s loss=0.0558 accuracy=0.9843 dev:loss=0.0759 dev:accuracy=0.9814
Epoch 10/10 1.4s loss=0.0502 accuracy=0.9860 dev:loss=0.0728 dev:accuracy=0.9806
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 1.9s loss=0.3025 accuracy=0.9152 dev:loss=0.1487 dev:accuracy=0.9582
Epoch  2/10 2.0s loss=0.1349 accuracy=0.9601 dev:loss=0.1003 dev:accuracy=0.9724
Epoch  3/10 2.0s loss=0.0909 accuracy=0.9724 dev:loss=0.0893 dev:accuracy=0.9756
Epoch  4/10 2.0s loss=0.0686 accuracy=0.9797 dev:loss=0.0879 dev:accuracy=0.9742
Epoch  5/10 2.0s loss=0.0542 accuracy=0.9838 dev:loss=0.0755 dev:accuracy=0.9782
Epoch  6/10 2.0s loss=0.0434 accuracy=0.9873 dev:loss=0.0781 dev:accuracy=0.9786
Epoch  7/10 2.0s loss=0.0344 accuracy=0.9900 dev:loss=0.0735 dev:accuracy=0.9796
Epoch  8/10 2.1s loss=0.0280 accuracy=0.9913 dev:loss=0.0746 dev:accuracy=0.9800
Epoch  9/10 2.0s loss=0.0225 accuracy=0.9934 dev:loss=0.0768 dev:accuracy=0.9814
Epoch 10/10 2.1s loss=0.0189 accuracy=0.9947 dev:loss=0.0838 dev:accuracy=0.9780
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 2.1s loss=0.2333 accuracy=0.9297 dev:loss=0.1618 dev:accuracy=0.9508
Epoch  2/10 2.4s loss=0.1456 accuracy=0.9569 dev:loss=0.1718 dev:accuracy=0.9600
Epoch  3/10 2.5s loss=0.1257 accuracy=0.9637 dev:loss=0.1653 dev:accuracy=0.9626
Epoch  4/10 2.5s loss=0.1128 accuracy=0.9679 dev:loss=0.1789 dev:accuracy=0.9604
Epoch  5/10 2.5s loss=0.1013 accuracy=0.9718 dev:loss=0.1316 dev:accuracy=0.9684
Epoch  6/10 2.6s loss=0.0992 accuracy=0.9729 dev:loss=0.1425 dev:accuracy=0.9642
Epoch  7/10 2.6s loss=0.0963 accuracy=0.9750 dev:loss=0.1814 dev:accuracy=0.9702
Epoch  8/10 2.8s loss=0.0969 accuracy=0.9759 dev:loss=0.1727 dev:accuracy=0.9712
Epoch  9/10 2.8s loss=0.0833 accuracy=0.9786 dev:loss=0.1854 dev:accuracy=0.9666
Epoch 10/10 2.9s loss=0.0808 accuracy=0.9796 dev:loss=0.1904 dev:accuracy=0.9710
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 2.3s lr=0.0090 loss=0.2329 accuracy=0.9295 dev:loss=0.1592 dev:accuracy=0.9542
Epoch  2/10 2.6s lr=0.0080 loss=0.1313 accuracy=0.9611 dev:loss=0.1211 dev:accuracy=0.9674
Epoch  3/10 2.6s lr=0.0070 loss=0.0983 accuracy=0.9696 dev:loss=0.1034 dev:accuracy=0.9734
Epoch  4/10 2.5s lr=0.0060 loss=0.0713 accuracy=0.9784 dev:loss=0.1250 dev:accuracy=0.9690
Epoch  5/10 2.6s lr=0.0051 loss=0.0557 accuracy=0.9825 dev:loss=0.1086 dev:accuracy=0.9748
Epoch  6/10 2.5s lr=0.0041 loss=0.0414 accuracy=0.9867 dev:loss=0.0983 dev:accuracy=0.9776
Epoch  7/10 2.5s lr=0.0031 loss=0.0246 accuracy=0.9921 dev:loss=0.1009 dev:accuracy=0.9782
Epoch  8/10 2.5s lr=0.0021 loss=0.0144 accuracy=0.9955 dev:loss=0.0996 dev:accuracy=0.9798
Epoch  9/10 2.5s lr=0.0011 loss=0.0072 accuracy=0.9979 dev:loss=0.0999 dev:accuracy=0.9800
Epoch 10/10 2.5s lr=1.00e-04 loss=0.0039 accuracy=0.9993 dev:loss=0.0985 dev:accuracy=0.9812
Next learning rate to be used: 0.0001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 2.1s lr=0.0079 loss=0.2235 accuracy=0.9331 dev:loss=0.1471 dev:accuracy=0.9584
Epoch  2/10 2.4s lr=0.0063 loss=0.1151 accuracy=0.9654 dev:loss=0.1097 dev:accuracy=0.9706
Epoch  3/10 2.4s lr=0.0050 loss=0.0782 accuracy=0.9757 dev:loss=0.1059 dev:accuracy=0.9748
Epoch  4/10 2.4s lr=0.0040 loss=0.0521 accuracy=0.9839 dev:loss=0.0984 dev:accuracy=0.9720
Epoch  5/10 2.5s lr=0.0032 loss=0.0366 accuracy=0.9879 dev:loss=0.1046 dev:accuracy=0.9764
Epoch  6/10 2.5s lr=0.0025 loss=0.0235 accuracy=0.9921 dev:loss=0.0965 dev:accuracy=0.9798
Epoch  7/10 2.5s lr=0.0020 loss=0.0144 accuracy=0.9954 dev:loss=0.0914 dev:accuracy=0.9810
Epoch  8/10 2.4s lr=0.0016 loss=0.0101 accuracy=0.9970 dev:loss=0.0924 dev:accuracy=0.9808
Epoch  9/10 2.4s lr=0.0013 loss=0.0057 accuracy=0.9986 dev:loss=0.1007 dev:accuracy=0.9820
Epoch 10/10 2.5s lr=0.0010 loss=0.0038 accuracy=0.9992 dev:loss=0.0926 dev:accuracy=0.9832
Next learning rate to be used: 0.001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch  1/10 2.1s lr=0.0098 loss=0.2362 accuracy=0.9288 dev:loss=0.1563 dev:accuracy=0.9556
Epoch  2/10 2.5s lr=0.0091 loss=0.1340 accuracy=0.9605 dev:loss=0.1450 dev:accuracy=0.9652
Epoch  3/10 2.5s lr=0.0080 loss=0.1088 accuracy=0.9688 dev:loss=0.1465 dev:accuracy=0.9612
Epoch  4/10 2.5s lr=0.0066 loss=0.0774 accuracy=0.9767 dev:loss=0.1184 dev:accuracy=0.9706
Epoch  5/10 2.6s lr=0.0050 loss=0.0569 accuracy=0.9823 dev:loss=0.1140 dev:accuracy=0.9762
Epoch  6/10 2.5s lr=0.0035 loss=0.0381 accuracy=0.9876 dev:loss=0.1166 dev:accuracy=0.9770
Epoch  7/10 2.5s lr=0.0021 loss=0.0195 accuracy=0.9939 dev:loss=0.1022 dev:accuracy=0.9800
Epoch  8/10 2.6s lr=0.0010 loss=0.0097 accuracy=0.9972 dev:loss=0.1059 dev:accuracy=0.9808
Epoch  9/10 2.6s lr=0.0003 loss=0.0055 accuracy=0.9989 dev:loss=0.1073 dev:accuracy=0.9792
Epoch 10/10 2.6s lr=1.00e-04 loss=0.0040 accuracy=0.9993 dev:loss=0.1071 dev:accuracy=0.9792
Next learning rate to be used: 0.0001
```
#### Examples End:
