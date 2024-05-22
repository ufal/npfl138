### Assignment: learning_to_learn
#### Date: Deadline: Jun 28, 22:00
#### Points: 4 points
#### Tests: learning_to_learn_tests
#### Examples: learning_to_learn_examples

Implement a simple variant of learning-to-learn architecture using the
[learning_to_learn.py](https://github.com/ufal/npfl138/tree/master/labs/14/learning_to_learn.py)
template. Utilizing the [Omniglot dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/omniglot_demo.html)
loadable using the [omniglot_dataset.py](https://github.com/ufal/npfl138/tree/master/labs/14/omniglot_dataset.py)
module, the goal is to learn to classify a
[sequence of images using a custom hierarchy](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/learning_to_learn_demo.html)
by employing external memory.

The input image sequences consist of `args.classes` randomly chosen Omniglot
classes, each class being assigned a randomly chosen label. For every chosen
class, `args.images_per_class` images are randomly selected. Apart from the
images, the input contain the random labels one step after the corresponding
images (with the first label being -1). The gold outputs are also the labels,
but without the one-step offset.

The input images should be passed through a CNN feature extraction module
and then processed using memory augmented LSTM controller; the external memory
contains enough memory cells, each with `args.cell_size` units. In each step,
the controller emits:
- `args.read_heads` read keys, each used to perform a read from memory as
  a weighted combination of cells according to the softmax of cosine
  similarities of the read key and the memory cells;
- a write value, which is prepended to the memory (dropping the last cell).

#### Tests Start: learning_to_learn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --classes=2`
```
Epoch 1/3 acc: 0.5001 - acc01: 0.4920 - acc02: 0.5172 - acc05: 0.4828 - acc10: 0.4844 - loss: 0.8351 - val_acc: 0.5409 - val_acc01: 0.7469 - val_acc02: 0.6438 - val_acc05: 0.4563 - val_acc10: 0.4719 - val_loss: 0.6917
Epoch 2/3 acc: 0.5116 - acc01: 0.5288 - acc02: 0.5545 - acc05: 0.5041 - acc10: 0.4829 - loss: 0.6987 - val_acc: 0.5516 - val_acc01: 0.7281 - val_acc02: 0.6281 - val_acc05: 0.5281 - val_acc10: 0.4812 - val_loss: 0.6911
Epoch 3/3 acc: 0.5074 - acc01: 0.5530 - acc02: 0.4786 - acc05: 0.5309 - acc10: 0.5526 - loss: 0.6969 - val_acc: 0.5544 - val_acc01: 0.7500 - val_acc02: 0.6187 - val_acc05: 0.5250 - val_acc10: 0.5312 - val_loss: 0.6903
```

2. `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --read_heads=2 --classes=5`
```
Epoch 1/3 acc: 0.2060 - acc01: 0.2127 - acc02: 0.2062 - acc05: 0.2097 - acc10: 0.1997 - loss: 1.6998 - val_acc: 0.2165 - val_acc01: 0.2750 - val_acc02: 0.2338 - val_acc05: 0.2100 - val_acc10: 0.2062 - val_loss: 1.6089
Epoch 2/3 acc: 0.2090 - acc01: 0.1978 - acc02: 0.2225 - acc05: 0.2155 - acc10: 0.2173 - loss: 1.6191 - val_acc: 0.2170 - val_acc01: 0.2675 - val_acc02: 0.2362 - val_acc05: 0.2125 - val_acc10: 0.2075 - val_loss: 1.6082
Epoch 3/3 acc: 0.2068 - acc01: 0.2096 - acc02: 0.2122 - acc05: 0.2125 - acc10: 0.2114 - loss: 1.6121 - val_acc: 0.2170 - val_acc01: 0.3375 - val_acc02: 0.2425 - val_acc05: 0.2025 - val_acc10: 0.1850 - val_loss: 1.6073
```
#### Tests End:
#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 learning_to_learn.py --epochs=50 --classes=2`
```
Epoch 1/50 acc: 0.5564 - acc01: 0.6674 - acc02: 0.5894 - acc05: 0.5423 - acc10: 0.5232 - loss: 0.6861 - val_acc: 0.6855 - val_acc01: 0.5710 - val_acc02: 0.7170 - val_acc05: 0.6725 - val_acc10: 0.7195 - val_loss: 0.5617
Epoch 2/50 acc: 0.7951 - acc01: 0.5817 - acc02: 0.7258 - acc05: 0.8191 - acc10: 0.8554 - loss: 0.3980 - val_acc: 0.8642 - val_acc01: 0.6505 - val_acc02: 0.7985 - val_acc05: 0.8975 - val_acc10: 0.9175 - val_loss: 0.2857
Epoch 3/50 acc: 0.8888 - acc01: 0.6122 - acc02: 0.8280 - acc05: 0.9253 - acc10: 0.9383 - loss: 0.2333 - val_acc: 0.8961 - val_acc01: 0.6235 - val_acc02: 0.8815 - val_acc05: 0.9390 - val_acc10: 0.9405 - val_loss: 0.2306
Epoch 4/50 acc: 0.9115 - acc01: 0.6406 - acc02: 0.8874 - acc05: 0.9486 - acc10: 0.9515 - loss: 0.1882 - val_acc: 0.9091 - val_acc01: 0.6515 - val_acc02: 0.8975 - val_acc05: 0.9490 - val_acc10: 0.9560 - val_loss: 0.1991
Epoch 5/50 acc: 0.9236 - acc01: 0.6606 - acc02: 0.9097 - acc05: 0.9604 - acc10: 0.9636 - loss: 0.1610 - val_acc: 0.9189 - val_acc01: 0.6575 - val_acc02: 0.9140 - val_acc05: 0.9560 - val_acc10: 0.9610 - val_loss: 0.1772
Epoch 6/50 acc: 0.9338 - acc01: 0.6729 - acc02: 0.9191 - acc05: 0.9650 - acc10: 0.9752 - loss: 0.1401 - val_acc: 0.9233 - val_acc01: 0.6740 - val_acc02: 0.9105 - val_acc05: 0.9565 - val_acc10: 0.9640 - val_loss: 0.1707
Epoch 7/50 acc: 0.9371 - acc01: 0.6712 - acc02: 0.9240 - acc05: 0.9708 - acc10: 0.9767 - loss: 0.1287 - val_acc: 0.9267 - val_acc01: 0.6680 - val_acc02: 0.9145 - val_acc05: 0.9570 - val_acc10: 0.9695 - val_loss: 0.1606
Epoch 8/50 acc: 0.9407 - acc01: 0.6732 - acc02: 0.9320 - acc05: 0.9756 - acc10: 0.9804 - loss: 0.1220 - val_acc: 0.9240 - val_acc01: 0.6495 - val_acc02: 0.9115 - val_acc05: 0.9595 - val_acc10: 0.9690 - val_loss: 0.1661
Epoch 9/50 acc: 0.9444 - acc01: 0.6851 - acc02: 0.9396 - acc05: 0.9762 - acc10: 0.9814 - loss: 0.1138 - val_acc: 0.9249 - val_acc01: 0.6590 - val_acc02: 0.9215 - val_acc05: 0.9580 - val_acc10: 0.9680 - val_loss: 0.1611
Epoch 10/50 acc: 0.9463 - acc01: 0.6877 - acc02: 0.9448 - acc05: 0.9789 - acc10: 0.9816 - loss: 0.1100 - val_acc: 0.9305 - val_acc01: 0.6615 - val_acc02: 0.9100 - val_acc05: 0.9650 - val_acc10: 0.9735 - val_loss: 0.1489
```
- `python3 learning_to_learn.py --epochs=50 --read_heads=2 --classes=5`
```
Epoch 1/50 acc: 0.2264 - acc01: 0.3060 - acc02: 0.2409 - acc05: 0.2182 - acc10: 0.2074 - loss: 1.6064 - val_acc: 0.3574 - val_acc01: 0.4056 - val_acc02: 0.3502 - val_acc05: 0.3428 - val_acc10: 0.3724 - val_loss: 1.3996
Epoch 2/50 acc: 0.5282 - acc01: 0.3375 - acc02: 0.4318 - acc05: 0.5459 - acc10: 0.6089 - loss: 1.0798 - val_acc: 0.6883 - val_acc01: 0.2558 - val_acc02: 0.5404 - val_acc05: 0.7600 - val_acc10: 0.7950 - val_loss: 0.7543
Epoch 3/50 acc: 0.7441 - acc01: 0.2457 - acc02: 0.5913 - acc05: 0.8241 - acc10: 0.8526 - loss: 0.6173 - val_acc: 0.7267 - val_acc01: 0.2368 - val_acc02: 0.5708 - val_acc05: 0.8094 - val_acc10: 0.8332 - val_loss: 0.6701
Epoch 4/50 acc: 0.7873 - acc01: 0.2615 - acc02: 0.6476 - acc05: 0.8705 - acc10: 0.8898 - loss: 0.5152 - val_acc: 0.7437 - val_acc01: 0.2796 - val_acc02: 0.6208 - val_acc05: 0.8090 - val_acc10: 0.8478 - val_loss: 0.6510
Epoch 5/50 acc: 0.8102 - acc01: 0.2724 - acc02: 0.6842 - acc05: 0.8954 - acc10: 0.9120 - loss: 0.4595 - val_acc: 0.7537 - val_acc01: 0.2778 - val_acc02: 0.6162 - val_acc05: 0.8280 - val_acc10: 0.8538 - val_loss: 0.6353
Epoch 6/50 acc: 0.8281 - acc01: 0.2776 - acc02: 0.7132 - acc05: 0.9124 - acc10: 0.9264 - loss: 0.4149 - val_acc: 0.7795 - val_acc01: 0.2640 - val_acc02: 0.6494 - val_acc05: 0.8592 - val_acc10: 0.8896 - val_loss: 0.5767
Epoch 7/50 acc: 0.8470 - acc01: 0.2946 - acc02: 0.7413 - acc05: 0.9308 - acc10: 0.9413 - loss: 0.3716 - val_acc: 0.8024 - val_acc01: 0.2882 - val_acc02: 0.6932 - val_acc05: 0.8784 - val_acc10: 0.9102 - val_loss: 0.5267
Epoch 8/50 acc: 0.8575 - acc01: 0.3049 - acc02: 0.7636 - acc05: 0.9385 - acc10: 0.9495 - loss: 0.3428 - val_acc: 0.8047 - val_acc01: 0.3032 - val_acc02: 0.7076 - val_acc05: 0.8692 - val_acc10: 0.9136 - val_loss: 0.5212
Epoch 9/50 acc: 0.8664 - acc01: 0.3124 - acc02: 0.7829 - acc05: 0.9461 - acc10: 0.9578 - loss: 0.3208 - val_acc: 0.8147 - val_acc01: 0.3038 - val_acc02: 0.7232 - val_acc05: 0.8860 - val_acc10: 0.9148 - val_loss: 0.4991
Epoch 10/50 acc: 0.8710 - acc01: 0.3129 - acc02: 0.7986 - acc05: 0.9482 - acc10: 0.9607 - loss: 0.3089 - val_acc: 0.8188 - val_acc01: 0.3148 - val_acc02: 0.7182 - val_acc05: 0.8888 - val_acc10: 0.9150 - val_loss: 0.4916
```
#### Examples End:
