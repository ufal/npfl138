### Assignment: mnist_ensemble
#### Date: Deadline: Mar 19, 22:00
#### Points: 2 points
#### Tests: mnist_ensemble_tests
#### Examples: mnist_ensemble_examples

Your goal in this assignment is to implement ensembling of
classification models by averaging their predicted probability distributions.
The [mnist_ensemble.py](https://github.com/ufal/npfl138/tree/master/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the development set.

#### Tests Start: mnist_ensemble_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_ensemble.py --epochs=1 --models=5`
```
Model 1, individual accuracy 96.08, ensemble accuracy 96.08
Model 2, individual accuracy 96.18, ensemble accuracy 96.48
Model 3, individual accuracy 96.02, ensemble accuracy 96.58
Model 4, individual accuracy 95.94, ensemble accuracy 96.64
Model 5, individual accuracy 96.14, ensemble accuracy 96.66
```

2. `python3 mnist_ensemble.py --epochs=1 --models=5 --hidden_layer_size=200`
```
Model 1, individual accuracy 96.58, ensemble accuracy 96.58
Model 2, individual accuracy 96.70, ensemble accuracy 96.80
Model 3, individual accuracy 96.70, ensemble accuracy 97.04
Model 4, individual accuracy 96.96, ensemble accuracy 97.14
Model 5, individual accuracy 96.76, ensemble accuracy 97.12
```
#### Tests End:
#### Examples Start: mnist_ensemble_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_ensemble.py --models=5`
```
Model 1, individual accuracy 97.76, ensemble accuracy 97.76
Model 2, individual accuracy 97.90, ensemble accuracy 98.08
Model 3, individual accuracy 97.92, ensemble accuracy 98.30
Model 4, individual accuracy 98.02, ensemble accuracy 98.36
Model 5, individual accuracy 97.86, ensemble accuracy 98.38
```

- `python3 mnist_ensemble.py --models=5 --hidden_layer_size=200`
```
Model 1, individual accuracy 98.10, ensemble accuracy 98.10
Model 2, individual accuracy 98.20, ensemble accuracy 98.42
Model 3, individual accuracy 97.90, ensemble accuracy 98.44
Model 4, individual accuracy 97.96, ensemble accuracy 98.46
Model 5, individual accuracy 97.90, ensemble accuracy 98.58
```
#### Examples End:
