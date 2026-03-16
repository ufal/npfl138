### Assignment: torch_dataset
#### Date: Deadline: Mar 25, 22:00
#### Points: 2 points
#### Tests: torch_dataset_tests

In this assignment you will familiarize yourself with `torch.utils.data`,
which is a PyTorch way of constructing training datasets. If you want,
you can read the [Dataset and DataLoaders
tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

The goal of this assignment is to start with the
[torch_dataset.py](https://github.com/ufal/npfl138/tree/master/labs/04/torch_dataset.py)
template and implement a simple image augmentation preprocessing.
The template also shows you how to use the
[npfl138.TransformedDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/transformed_dataset/)
module.

#### Tests Start: torch_dataset_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 torch_dataset.py --recodex --epochs=1 --batch_size=100`
```
Epoch 1/1 2.0s loss=2.1335 accuracy=0.2026 dev:loss=1.9201 dev:accuracy=0.2750
```

2. `python3 torch_dataset.py --recodex --epochs=1 --batch_size=50 --augment`
```
Epoch 1/1 2.6s loss=2.1103 accuracy=0.2054 dev:loss=1.9402 dev:accuracy=0.2700
```
#### Tests End:
