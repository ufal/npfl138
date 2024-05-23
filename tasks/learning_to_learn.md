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
Epoch 1/3 acc: 0.5001 - acc01: 0.4920 - acc02: 0.5172 - acc05: 0.4828 - acc10: 0.4844 - loss: 0.8351 - val_acc: 0.5406 - val_acc01: 0.7469 - val_acc02: 0.6406 - val_acc05: 0.4563 - val_acc10: 0.4719 - val_loss: 0.6917
Epoch 2/3 acc: 0.5116 - acc01: 0.5288 - acc02: 0.5545 - acc05: 0.5041 - acc10: 0.4829 - loss: 0.6987 - val_acc: 0.5516 - val_acc01: 0.7250 - val_acc02: 0.6281 - val_acc05: 0.5281 - val_acc10: 0.4812 - val_loss: 0.6911
Epoch 3/3 acc: 0.5074 - acc01: 0.5530 - acc02: 0.4786 - acc05: 0.5309 - acc10: 0.5526 - loss: 0.6969 - val_acc: 0.5544 - val_acc01: 0.7500 - val_acc02: 0.6187 - val_acc05: 0.5250 - val_acc10: 0.5312 - val_loss: 0.6903
```

2. `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --read_heads=2 --classes=5`
```
Epoch 1/3 acc: 0.2060 - acc01: 0.2127 - acc02: 0.2062 - acc05: 0.2097 - acc10: 0.1997 - loss: 1.6998 - val_acc: 0.2165 - val_acc01: 0.2750 - val_acc02: 0.2338 - val_acc05: 0.2100 - val_acc10: 0.2062 - val_loss: 1.6089
Epoch 2/3 acc: 0.2088 - acc01: 0.1978 - acc02: 0.2190 - acc05: 0.2155 - acc10: 0.2173 - loss: 1.6191 - val_acc: 0.2176 - val_acc01: 0.2663 - val_acc02: 0.2362 - val_acc05: 0.2125 - val_acc10: 0.2100 - val_loss: 1.6082
Epoch 3/3 acc: 0.2067 - acc01: 0.2096 - acc02: 0.2122 - acc05: 0.2125 - acc10: 0.2114 - loss: 1.6121 - val_acc: 0.2171 - val_acc01: 0.3375 - val_acc02: 0.2425 - val_acc05: 0.2025 - val_acc10: 0.1850 - val_loss: 1.6073
```
#### Tests End:
#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 learning_to_learn.py --epochs=50 --classes=2`
```
Epoch 1/50 acc: 0.5611 - acc01: 0.6706 - acc02: 0.5922 - acc05: 0.5470 - acc10: 0.5286 - loss: 0.6838 - val_acc: 0.7102 - val_acc01: 0.5875 - val_acc02: 0.6625 - val_acc05: 0.7120 - val_acc10: 0.7850 - val_loss: 0.5361
Epoch 2/50 acc: 0.8047 - acc01: 0.6067 - acc02: 0.7412 - acc05: 0.8309 - acc10: 0.8573 - loss: 0.3806 - val_acc: 0.8364 - val_acc01: 0.6295 - val_acc02: 0.8075 - val_acc05: 0.8750 - val_acc10: 0.8790 - val_loss: 0.3403
Epoch 3/50 acc: 0.8885 - acc01: 0.6261 - acc02: 0.8607 - acc05: 0.9249 - acc10: 0.9296 - loss: 0.2362 - val_acc: 0.8612 - val_acc01: 0.5575 - val_acc02: 0.8320 - val_acc05: 0.9015 - val_acc10: 0.9170 - val_loss: 0.3095
Epoch 4/50 acc: 0.9165 - acc01: 0.6401 - acc02: 0.8890 - acc05: 0.9506 - acc10: 0.9627 - loss: 0.1759 - val_acc: 0.9072 - val_acc01: 0.6485 - val_acc02: 0.8875 - val_acc05: 0.9415 - val_acc10: 0.9605 - val_loss: 0.2042
Epoch 5/50 acc: 0.9327 - acc01: 0.6691 - acc02: 0.9197 - acc05: 0.9667 - acc10: 0.9732 - loss: 0.1437 - val_acc: 0.8991 - val_acc01: 0.5850 - val_acc02: 0.9005 - val_acc05: 0.9355 - val_acc10: 0.9525 - val_loss: 0.2272
Epoch 10/50 acc: 0.9489 - acc01: 0.6942 - acc02: 0.9508 - acc05: 0.9790 - acc10: 0.9824 - loss: 0.1038 - val_acc: 0.9100 - val_acc01: 0.6355 - val_acc02: 0.8875 - val_acc05: 0.9500 - val_acc10: 0.9680 - val_loss: 0.1962
Epoch 20/50 acc: 0.9585 - acc01: 0.7080 - acc02: 0.9676 - acc05: 0.9882 - acc10: 0.9917 - loss: 0.0788 - val_acc: 0.9362 - val_acc01: 0.6935 - val_acc02: 0.9300 - val_acc05: 0.9675 - val_acc10: 0.9785 - val_loss: 0.1425
Epoch 50/50 acc: 0.9663 - acc01: 0.7207 - acc02: 0.9819 - acc05: 0.9954 - acc10: 0.9961 - loss: 0.0573 - val_acc: 0.9486 - val_acc01: 0.6915 - val_acc02: 0.9550 - val_acc05: 0.9790 - val_acc10: 0.9865 - val_loss: 0.1137
```
- `python3 learning_to_learn.py --epochs=50 --read_heads=2 --classes=5`
```
Epoch 1/50 acc: 0.2279 - acc01: 0.3091 - acc02: 0.2439 - acc05: 0.2195 - acc10: 0.2099 - loss: 1.6053 - val_acc: 0.3467 - val_acc01: 0.4224 - val_acc02: 0.3386 - val_acc05: 0.3262 - val_acc10: 0.3548 - val_loss: 1.4456
Epoch 2/50 acc: 0.5093 - acc01: 0.3486 - acc02: 0.4208 - acc05: 0.5255 - acc10: 0.5849 - loss: 1.1036 - val_acc: 0.6941 - val_acc01: 0.2430 - val_acc02: 0.5560 - val_acc05: 0.7634 - val_acc10: 0.8052 - val_loss: 0.7470
Epoch 3/50 acc: 0.7590 - acc01: 0.2540 - acc02: 0.6111 - acc05: 0.8375 - acc10: 0.8680 - loss: 0.5842 - val_acc: 0.7268 - val_acc01: 0.2454 - val_acc02: 0.5834 - val_acc05: 0.8058 - val_acc10: 0.8350 - val_loss: 0.6883
Epoch 4/50 acc: 0.8060 - acc01: 0.2715 - acc02: 0.6700 - acc05: 0.8898 - acc10: 0.9108 - loss: 0.4713 - val_acc: 0.7557 - val_acc01: 0.2686 - val_acc02: 0.6314 - val_acc05: 0.8292 - val_acc10: 0.8602 - val_loss: 0.6230
Epoch 5/50 acc: 0.8264 - acc01: 0.2786 - acc02: 0.7133 - acc05: 0.9115 - acc10: 0.9269 - loss: 0.4206 - val_acc: 0.7596 - val_acc01: 0.2610 - val_acc02: 0.6358 - val_acc05: 0.8386 - val_acc10: 0.8612 - val_loss: 0.6250
Epoch 10/50 acc: 0.8714 - acc01: 0.3127 - acc02: 0.8093 - acc05: 0.9482 - acc10: 0.9591 - loss: 0.3091 - val_acc: 0.8045 - val_acc01: 0.2998 - val_acc02: 0.7284 - val_acc05: 0.8680 - val_acc10: 0.8962 - val_loss: 0.5422
Epoch 20/50 acc: 0.9008 - acc01: 0.3432 - acc02: 0.8962 - acc05: 0.9705 - acc10: 0.9762 - loss: 0.2338 - val_acc: 0.8096 - val_acc01: 0.3052 - val_acc02: 0.7614 - val_acc05: 0.8754 - val_acc10: 0.8976 - val_loss: 0.5662
Epoch 50/50 acc: 0.9236 - acc01: 0.3911 - acc02: 0.9526 - acc05: 0.9867 - acc10: 0.9899 - loss: 0.1733 - val_acc: 0.8387 - val_acc01: 0.3372 - val_acc02: 0.8224 - val_acc05: 0.8978 - val_acc10: 0.9232 - val_loss: 0.5138
```
#### Examples End:
