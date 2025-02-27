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
Epoch 1/1 1.2s train_loss=0.8300 train_accuracy=0.7960 dev_loss=0.3780 dev_accuracy=0.9060
```

2. `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch 1/1 1.2s train_loss=0.3731 train_accuracy=0.8952 dev_loss=0.1912 dev_accuracy=0.9472
```

3. `python3 mnist_training.py --epochs=1 --optimizer=SGD --learning_rate=0.1`
```
Epoch 1/1 1.1s train_loss=0.3660 train_accuracy=0.8970 dev_loss=0.1945 dev_accuracy=0.9460
```

4. `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.001`
```
Epoch 1/1 1.5s train_loss=0.3025 train_accuracy=0.9152 dev_loss=0.1487 dev_accuracy=0.9582
```

5. `python3 mnist_training.py --epochs=1 --optimizer=Adam --learning_rate=0.01`
```
Epoch 1/1 1.6s train_loss=0.2333 train_accuracy=0.9297 dev_loss=0.1618 dev_accuracy=0.9508
```

6. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch 1/2 1.6s train_loss=0.2162 train_lr=0.0050 train_accuracy=0.9341 dev_loss=0.1150 dev_accuracy=0.9658
Epoch 2/2 1.9s train_loss=0.0790 train_lr=0.0001 train_accuracy=0.9759 dev_loss=0.0739 dev_accuracy=0.9778
Next learning rate to be used: 0.0001
```

7. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch 1/2 1.6s train_loss=0.2022 train_lr=0.0032 train_accuracy=0.9383 dev_loss=0.0989 dev_accuracy=0.9746
Epoch 2/2 1.8s train_loss=0.0748 train_lr=0.0010 train_accuracy=0.9769 dev_loss=0.0777 dev_accuracy=0.9790
Next learning rate to be used: 0.001
```

8. `python3 mnist_training.py --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch 1/2 1.7s train_loss=0.2192 train_lr=0.0050 train_accuracy=0.9333 dev_loss=0.1155 dev_accuracy=0.9680
Epoch 2/2 1.9s train_loss=0.0720 train_lr=0.0001 train_accuracy=0.9776 dev_loss=0.0765 dev_accuracy=0.9790
Next learning rate to be used: 0.0001
```
#### Tests End:
#### Examples Start: mnist_training_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01`
```
Epoch  1/10 1.1s train_loss=0.8300 train_accuracy=0.7960 dev_loss=0.3780 dev_accuracy=0.9060
Epoch  2/10 1.1s train_loss=0.4088 train_accuracy=0.8892 dev_loss=0.2940 dev_accuracy=0.9208
Epoch  3/10 1.1s train_loss=0.3473 train_accuracy=0.9030 dev_loss=0.2585 dev_accuracy=0.9286
Epoch  4/10 1.1s train_loss=0.3144 train_accuracy=0.9116 dev_loss=0.2383 dev_accuracy=0.9352
Epoch  5/10 1.1s train_loss=0.2911 train_accuracy=0.9184 dev_loss=0.2230 dev_accuracy=0.9404
Epoch  6/10 1.1s train_loss=0.2729 train_accuracy=0.9235 dev_loss=0.2093 dev_accuracy=0.9432
Epoch  7/10 1.2s train_loss=0.2577 train_accuracy=0.9281 dev_loss=0.1993 dev_accuracy=0.9480
Epoch  8/10 1.1s train_loss=0.2442 train_accuracy=0.9316 dev_loss=0.1903 dev_accuracy=0.9510
Epoch  9/10 1.1s train_loss=0.2326 train_accuracy=0.9350 dev_loss=0.1828 dev_accuracy=0.9546
Epoch 10/10 1.1s train_loss=0.2222 train_accuracy=0.9379 dev_loss=0.1744 dev_accuracy=0.9546
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9`
```
Epoch  1/10 1.2s train_loss=0.3731 train_accuracy=0.8952 dev_loss=0.1912 dev_accuracy=0.9472
Epoch  2/10 1.3s train_loss=0.1942 train_accuracy=0.9437 dev_loss=0.1322 dev_accuracy=0.9662
Epoch  3/10 1.4s train_loss=0.1432 train_accuracy=0.9588 dev_loss=0.1137 dev_accuracy=0.9688
Epoch  4/10 1.4s train_loss=0.1148 train_accuracy=0.9674 dev_loss=0.0954 dev_accuracy=0.9744
Epoch  5/10 1.4s train_loss=0.0962 train_accuracy=0.9728 dev_loss=0.0914 dev_accuracy=0.9740
Epoch  6/10 1.4s train_loss=0.0824 train_accuracy=0.9767 dev_loss=0.0823 dev_accuracy=0.9772
Epoch  7/10 1.4s train_loss=0.0718 train_accuracy=0.9801 dev_loss=0.0806 dev_accuracy=0.9780
Epoch  8/10 1.4s train_loss=0.0640 train_accuracy=0.9817 dev_loss=0.0741 dev_accuracy=0.9800
Epoch  9/10 1.4s train_loss=0.0565 train_accuracy=0.9841 dev_loss=0.0775 dev_accuracy=0.9800
Epoch 10/10 1.4s train_loss=0.0509 train_accuracy=0.9861 dev_loss=0.0737 dev_accuracy=0.9788
```

- `python3 mnist_training.py --optimizer=SGD --learning_rate=0.1`
```
Epoch  1/10 1.2s train_loss=0.3660 train_accuracy=0.8970 dev_loss=0.1945 dev_accuracy=0.9460
Epoch  2/10 1.1s train_loss=0.1940 train_accuracy=0.9438 dev_loss=0.1320 dev_accuracy=0.9652
Epoch  3/10 1.1s train_loss=0.1433 train_accuracy=0.9588 dev_loss=0.1101 dev_accuracy=0.9696
Epoch  4/10 1.2s train_loss=0.1146 train_accuracy=0.9673 dev_loss=0.0941 dev_accuracy=0.9748
Epoch  5/10 1.2s train_loss=0.0949 train_accuracy=0.9735 dev_loss=0.0915 dev_accuracy=0.9754
Epoch  6/10 1.1s train_loss=0.0816 train_accuracy=0.9766 dev_loss=0.0804 dev_accuracy=0.9782
Epoch  7/10 1.1s train_loss=0.0714 train_accuracy=0.9800 dev_loss=0.0783 dev_accuracy=0.9792
Epoch  8/10 1.1s train_loss=0.0627 train_accuracy=0.9819 dev_loss=0.0734 dev_accuracy=0.9804
Epoch  9/10 1.1s train_loss=0.0558 train_accuracy=0.9843 dev_loss=0.0759 dev_accuracy=0.9814
Epoch 10/10 1.2s train_loss=0.0502 train_accuracy=0.9860 dev_loss=0.0728 dev_accuracy=0.9806
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.001`
```
Epoch  1/10 1.5s train_loss=0.3025 train_accuracy=0.9152 dev_loss=0.1487 dev_accuracy=0.9582
Epoch  2/10 1.6s train_loss=0.1349 train_accuracy=0.9601 dev_loss=0.1003 dev_accuracy=0.9724
Epoch  3/10 1.6s train_loss=0.0909 train_accuracy=0.9724 dev_loss=0.0893 dev_accuracy=0.9756
Epoch  4/10 1.6s train_loss=0.0686 train_accuracy=0.9797 dev_loss=0.0879 dev_accuracy=0.9742
Epoch  5/10 1.6s train_loss=0.0542 train_accuracy=0.9838 dev_loss=0.0755 dev_accuracy=0.9782
Epoch  6/10 1.6s train_loss=0.0434 train_accuracy=0.9873 dev_loss=0.0781 dev_accuracy=0.9786
Epoch  7/10 1.6s train_loss=0.0344 train_accuracy=0.9900 dev_loss=0.0735 dev_accuracy=0.9796
Epoch  8/10 1.7s train_loss=0.0280 train_accuracy=0.9913 dev_loss=0.0746 dev_accuracy=0.9800
Epoch  9/10 1.6s train_loss=0.0225 train_accuracy=0.9934 dev_loss=0.0768 dev_accuracy=0.9814
Epoch 10/10 1.6s train_loss=0.0189 train_accuracy=0.9947 dev_loss=0.0838 dev_accuracy=0.9780
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01`
```
Epoch  1/10 1.6s train_loss=0.2333 train_accuracy=0.9297 dev_loss=0.1618 dev_accuracy=0.9508
Epoch  2/10 1.9s train_loss=0.1456 train_accuracy=0.9569 dev_loss=0.1718 dev_accuracy=0.9600
Epoch  3/10 1.9s train_loss=0.1257 train_accuracy=0.9637 dev_loss=0.1653 dev_accuracy=0.9626
Epoch  4/10 1.9s train_loss=0.1128 train_accuracy=0.9679 dev_loss=0.1789 dev_accuracy=0.9604
Epoch  5/10 1.9s train_loss=0.1013 train_accuracy=0.9718 dev_loss=0.1316 dev_accuracy=0.9684
Epoch  6/10 2.0s train_loss=0.0992 train_accuracy=0.9729 dev_loss=0.1425 dev_accuracy=0.9642
Epoch  7/10 2.0s train_loss=0.0963 train_accuracy=0.9750 dev_loss=0.1814 dev_accuracy=0.9702
Epoch  8/10 2.0s train_loss=0.0969 train_accuracy=0.9759 dev_loss=0.1727 dev_accuracy=0.9712
Epoch  9/10 2.0s train_loss=0.0833 train_accuracy=0.9786 dev_loss=0.1854 dev_accuracy=0.9666
Epoch 10/10 2.0s train_loss=0.0808 train_accuracy=0.9796 dev_loss=0.1904 dev_accuracy=0.9710
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001`
```
Epoch  1/10 1.6s train_loss=0.2329 train_lr=0.0090 train_accuracy=0.9295 dev_loss=0.1592 dev_accuracy=0.9542
Epoch  2/10 1.9s train_loss=0.1313 train_lr=0.0080 train_accuracy=0.9611 dev_loss=0.1211 dev_accuracy=0.9674
Epoch  3/10 1.9s train_loss=0.0983 train_lr=0.0070 train_accuracy=0.9696 dev_loss=0.1034 dev_accuracy=0.9734
Epoch  4/10 1.9s train_loss=0.0713 train_lr=0.0060 train_accuracy=0.9784 dev_loss=0.1250 dev_accuracy=0.9690
Epoch  5/10 1.9s train_loss=0.0557 train_lr=0.0051 train_accuracy=0.9825 dev_loss=0.1086 dev_accuracy=0.9748
Epoch  6/10 1.9s train_loss=0.0414 train_lr=0.0041 train_accuracy=0.9867 dev_loss=0.0983 dev_accuracy=0.9776
Epoch  7/10 1.9s train_loss=0.0246 train_lr=0.0031 train_accuracy=0.9921 dev_loss=0.1009 dev_accuracy=0.9782
Epoch  8/10 1.9s train_loss=0.0144 train_lr=0.0021 train_accuracy=0.9955 dev_loss=0.0996 dev_accuracy=0.9798
Epoch  9/10 2.0s train_loss=0.0072 train_lr=0.0011 train_accuracy=0.9979 dev_loss=0.0999 dev_accuracy=0.9800
Epoch 10/10 1.9s train_loss=0.0039 train_lr=0.0001 train_accuracy=0.9993 dev_loss=0.0985 dev_accuracy=0.9812
Next learning rate to be used: 0.0001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001`
```
Epoch  1/10 1.6s train_loss=0.2235 train_lr=0.0079 train_accuracy=0.9331 dev_loss=0.1471 dev_accuracy=0.9584
Epoch  2/10 1.9s train_loss=0.1151 train_lr=0.0063 train_accuracy=0.9654 dev_loss=0.1097 dev_accuracy=0.9706
Epoch  3/10 1.9s train_loss=0.0782 train_lr=0.0050 train_accuracy=0.9757 dev_loss=0.1059 dev_accuracy=0.9748
Epoch  4/10 1.9s train_loss=0.0521 train_lr=0.0040 train_accuracy=0.9839 dev_loss=0.0984 dev_accuracy=0.9720
Epoch  5/10 1.9s train_loss=0.0366 train_lr=0.0032 train_accuracy=0.9879 dev_loss=0.1046 dev_accuracy=0.9764
Epoch  6/10 1.9s train_loss=0.0235 train_lr=0.0025 train_accuracy=0.9921 dev_loss=0.0965 dev_accuracy=0.9798
Epoch  7/10 1.9s train_loss=0.0144 train_lr=0.0020 train_accuracy=0.9954 dev_loss=0.0914 dev_accuracy=0.9810
Epoch  8/10 1.9s train_loss=0.0101 train_lr=0.0016 train_accuracy=0.9970 dev_loss=0.0924 dev_accuracy=0.9808
Epoch  9/10 1.9s train_loss=0.0057 train_lr=0.0013 train_accuracy=0.9986 dev_loss=0.1007 dev_accuracy=0.9820
Epoch 10/10 1.9s train_loss=0.0038 train_lr=0.0010 train_accuracy=0.9992 dev_loss=0.0926 dev_accuracy=0.9832
Next learning rate to be used: 0.001
```

- `python3 mnist_training.py --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001`
```
Epoch  1/10 1.6s train_loss=0.2362 train_lr=0.0098 train_accuracy=0.9288 dev_loss=0.1563 dev_accuracy=0.9556
Epoch  2/10 1.9s train_loss=0.1340 train_lr=0.0091 train_accuracy=0.9605 dev_loss=0.1450 dev_accuracy=0.9652
Epoch  3/10 1.9s train_loss=0.1088 train_lr=0.0080 train_accuracy=0.9688 dev_loss=0.1465 dev_accuracy=0.9612
Epoch  4/10 2.0s train_loss=0.0774 train_lr=0.0066 train_accuracy=0.9767 dev_loss=0.1184 dev_accuracy=0.9706
Epoch  5/10 1.9s train_loss=0.0569 train_lr=0.0050 train_accuracy=0.9823 dev_loss=0.1140 dev_accuracy=0.9762
Epoch  6/10 2.0s train_loss=0.0381 train_lr=0.0035 train_accuracy=0.9876 dev_loss=0.1166 dev_accuracy=0.9770
Epoch  7/10 1.9s train_loss=0.0195 train_lr=0.0021 train_accuracy=0.9939 dev_loss=0.1022 dev_accuracy=0.9800
Epoch  8/10 1.9s train_loss=0.0097 train_lr=0.0010 train_accuracy=0.9972 dev_loss=0.1059 dev_accuracy=0.9808
Epoch  9/10 1.9s train_loss=0.0055 train_lr=0.0003 train_accuracy=0.9989 dev_loss=0.1073 dev_accuracy=0.9792
Epoch 10/10 1.9s train_loss=0.0040 train_lr=0.0001 train_accuracy=0.9993 dev_loss=0.1071 dev_accuracy=0.9792
Next learning rate to be used: 0.0001
```
#### Examples End:
