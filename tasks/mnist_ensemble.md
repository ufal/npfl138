### Assignment: mnist_ensemble
#### Date: Deadline: Mar 19, 22:00
#### Points: 2 points
#### Tests: mnist_ensemble_tests
#### Examples: mnist_ensemble_examples

Your goal in this assignment is to implement model ensembling.
The [mnist_ensemble.py](https://github.com/ufal/npfl138/tree/past-2324/labs/03/mnist_ensemble.py)
template trains `args.models` individual models, and your goal is to perform
an ensemble of the first model, first two models, first three models, …, all
models, and evaluate their accuracy on the development set.

#### Tests Start: mnist_ensemble_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 mnist_ensemble.py --epochs=1 --models=5`
```
Model 1, individual accuracy 96.04, ensemble accuracy 96.04
Model 2, individual accuracy 96.28, ensemble accuracy 96.56
Model 3, individual accuracy 96.12, ensemble accuracy 96.58
Model 4, individual accuracy 95.92, ensemble accuracy 96.70
Model 5, individual accuracy 96.38, ensemble accuracy 96.72
```

2. `python3 mnist_ensemble.py --epochs=1 --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 96.46, ensemble accuracy 96.46
Model 2, individual accuracy 96.86, ensemble accuracy 96.88
Model 3, individual accuracy 96.54, ensemble accuracy 97.04
Model 4, individual accuracy 96.54, ensemble accuracy 97.06
Model 5, individual accuracy 96.82, ensemble accuracy 97.20
```
#### Tests End:
#### Examples Start: mnist_ensemble_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 mnist_ensemble.py --models=5`
```
Model 1, individual accuracy 97.82, ensemble accuracy 97.82
Model 2, individual accuracy 97.80, ensemble accuracy 98.08
Model 3, individual accuracy 98.02, ensemble accuracy 98.20
Model 4, individual accuracy 98.20, ensemble accuracy 98.28
Model 5, individual accuracy 97.64, ensemble accuracy 98.28
```

- `python3 mnist_ensemble.py --models=5 --hidden_layers=200`
```
Model 1, individual accuracy 98.12, ensemble accuracy 98.12
Model 2, individual accuracy 98.22, ensemble accuracy 98.42
Model 3, individual accuracy 98.26, ensemble accuracy 98.52
Model 4, individual accuracy 98.32, ensemble accuracy 98.62
Model 5, individual accuracy 97.98, ensemble accuracy 98.70
```
#### Examples End:
