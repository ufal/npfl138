### Assignment: sgd_manual
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: sgd_manual_tests
#### Examples: sgd_manual_examples

_**The template was updated on Feb 27, 17:30.** The original one did not shuffle
the training data. You do not need to redownload it, ReCodEx accepts both variants.
However, the Tests and Examples have been regenerated using the updated template._

The goal in this exercise is to extend your solution to the
[sgd_backpropagation](https://ufal.mff.cuni.cz/courses/npfl138/2425-summer#sgd_backpropagation)
assignment by **manually** computing the gradient.

While in this assignment we compute the gradient manually, we will nearly always
use the automatic differentiation. Therefore, the assignment is more of
a mathematical exercise than a real-world application. Furthermore, we will
compute the derivatives together on the Mar 06 practicals.

Start with the
[sgd_manual.py](https://github.com/ufal/npfl138/tree/master/labs/02/sgd_manual.py)
template, which is based on the
[sgd_backpropagation.py](https://github.com/ufal/npfl138/tree/master/labs/02/sgd_backpropagation.py)
one.

Note that ReCodEx disables the PyTorch automatic differentiation during
evaluation.

#### Tests Start: sgd_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sgd_manual.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.98
Dev accuracy after epoch 2 is 94.42
Test accuracy after epoch 2 is 92.72
```

2. `python3 sgd_manual.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.58
Dev accuracy after epoch 2 is 95.26
Test accuracy after epoch 2 is 93.75
```
#### Tests End:
#### Examples Start: sgd_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sgd_manual.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 92.98
Dev accuracy after epoch 2 is 94.42
Dev accuracy after epoch 3 is 94.68
Dev accuracy after epoch 4 is 95.08
Dev accuracy after epoch 5 is 95.28
Dev accuracy after epoch 6 is 95.20
Dev accuracy after epoch 7 is 95.52
Dev accuracy after epoch 8 is 95.32
Dev accuracy after epoch 9 is 95.66
Dev accuracy after epoch 10 is 95.84
Test accuracy after epoch 10 is 95.02
```

- `python3 sgd_manual.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.58
Dev accuracy after epoch 2 is 95.26
Dev accuracy after epoch 3 is 95.66
Dev accuracy after epoch 4 is 95.90
Dev accuracy after epoch 5 is 96.26
Dev accuracy after epoch 6 is 96.52
Dev accuracy after epoch 7 is 96.52
Dev accuracy after epoch 8 is 96.74
Dev accuracy after epoch 9 is 96.74
Dev accuracy after epoch 10 is 96.62
Test accuracy after epoch 10 is 95.84
```
#### Examples End:
