### Assignment: sgd_manual
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: sgd_manual_tests
#### Examples: sgd_manual_examples

The goal in this exercise is to extend your solution to the
[sgd_backpropagation](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#sgd_backpropagation)
assignment by **manually** computing the gradient.

While in this assignment we compute the gradient manually, we will nearly always
use the automatic differentiation. Therefore, the assignment is more of
a mathematical exercise than a real-world application. Furthermore, we will
compute the derivatives together on the Mar 06 practicals.

Start with the
[sgd_manual.py](https://github.com/ufal/npfl138/tree/master/labs/02/sgd_manual.py)
template, which is based on
[sgd_backpropagation.py](https://github.com/ufal/npfl138/tree/master/labs/02/sgd_backpropagation.py)
one. Be aware that these templates generates each a different output file.

Note that ReCodEx disables the PyTorch automatic differentiation during
evaluation.

#### Tests Start: sgd_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 sgd_manual.py --epochs=2 --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Test accuracy after epoch 2 is 93.15
```

2. `python3 sgd_manual.py --epochs=2 --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.64
Dev accuracy after epoch 2 is 94.80
Test accuracy after epoch 2 is 93.54
```
#### Tests End:
#### Examples Start: sgd_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 sgd_manual.py --batch_size=64 --hidden_layer=20 --learning_rate=0.1`
```
Dev accuracy after epoch 1 is 93.30
Dev accuracy after epoch 2 is 94.38
Dev accuracy after epoch 3 is 95.16
Dev accuracy after epoch 4 is 95.50
Dev accuracy after epoch 5 is 95.96
Dev accuracy after epoch 6 is 96.04
Dev accuracy after epoch 7 is 95.82
Dev accuracy after epoch 8 is 95.92
Dev accuracy after epoch 9 is 95.96
Dev accuracy after epoch 10 is 96.16
Test accuracy after epoch 10 is 95.26
```

- `python3 sgd_manual.py --batch_size=100 --hidden_layer=32 --learning_rate=0.2`
```
Dev accuracy after epoch 1 is 93.64
Dev accuracy after epoch 2 is 94.80
Dev accuracy after epoch 3 is 95.56
Dev accuracy after epoch 4 is 95.98
Dev accuracy after epoch 5 is 96.24
Dev accuracy after epoch 6 is 96.74
Dev accuracy after epoch 7 is 96.52
Dev accuracy after epoch 8 is 96.54
Dev accuracy after epoch 9 is 97.04
Dev accuracy after epoch 10 is 97.02
Test accuracy after epoch 10 is 96.16
```
#### Examples End:
