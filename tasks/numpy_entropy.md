### Assignment: numpy_entropy
#### Date: Deadline: Mar 05, 22:00
#### Points: 3 points
#### Tests: numpy_entropy_tests
#### Examples: numpy_entropy_examples

The goal of this exercise is to familiarize with Python, NumPy and ReCodEx
submission system. Start with the
[numpy_entropy.py](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy.py).

Load a file specified in `args.data_path`, whose lines consist of data points of our
dataset, and load a file specified in `args.model_path`, which describes a model probability distribution,
with each line being a tab-separated pair of _(data point, probability)_.

Then compute the following quantities using NumPy, and print them each on
a separate line rounded on two decimal places (or `inf` for positive infinity,
which happens when an element of data distribution has zero probability
under the model distribution):
- entropy _H(data distribution)_
- cross-entropy _H(data distribution, model distribution)_
- KL-divergence _D<sub>KL</sub>(data distribution, model distribution)_

Use natural logarithms to compute the entropies and the divergence.

#### Tests Start: numpy_entropy_tests

1. `python3 numpy_entropy.py --data_path` [numpy_entropy_data_1.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_1.txt) `--model_path` [numpy_entropy_model_1.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_1.txt)
```
Entropy: 0.96 nats
Crossentropy: 0.99 nats
KL divergence: 0.03 nats
```

2. `python3 numpy_entropy.py --data_path` [numpy_entropy_data_2.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_2.txt) `--model_path` [numpy_entropy_model_2.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_2.txt)
```
Entropy: 0.96 nats
Crossentropy: inf nats
KL divergence: inf nats
```

- The last three tests use data available only in ReCodEx. They are analogous
  to the [numpy_entropy_data_3.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_3.txt)
  [numpy_entropy_model_3.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_3.txt)
  and [numpy_entropy_data_4.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_4.txt)
  [numpy_entropy_model_4.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_4.txt),
  but are generated with different random seeds.
#### Tests End:
#### Examples Start: numpy_entropy_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 numpy_entropy.py --data_path` [numpy_entropy_data_1.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_1.txt) `--model_path` [numpy_entropy_model_1.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_1.txt)
```
Entropy: 0.96 nats
Crossentropy: 0.99 nats
KL divergence: 0.03 nats
```

- `python3 numpy_entropy.py --data_path` [numpy_entropy_data_2.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_2.txt) `--model_path` [numpy_entropy_model_2.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_2.txt)
```
Entropy: 0.96 nats
Crossentropy: inf nats
KL divergence: inf nats
```

- `python3 numpy_entropy.py --data_path` [numpy_entropy_data_3.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_3.txt) `--model_path` [numpy_entropy_model_3.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_3.txt)
```
Entropy: 4.15 nats
Crossentropy: 4.23 nats
KL divergence: 0.08 nats
```

- `python3 numpy_entropy.py --data_path` [numpy_entropy_data_4.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_data_4.txt) `--model_path` [numpy_entropy_model_4.txt](https://github.com/ufal/npfl138/tree/master/labs/01/numpy_entropy_model_4.txt)
```
Entropy: 4.99 nats
Crossentropy: 5.03 nats
KL divergence: 0.04 nats
```
#### Examples End:
