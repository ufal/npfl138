### Assignment: torch_dataset
#### Date: Deadline: Mar 26, 22:00
#### Points: 2 points
#### Tests: torch_dataset_tests

In this assignment you will familiarize yourselves with `torch.utils.data`,
which is a PyTorch way of constructing training datasets. If you want,
you can read the [Dataset and DataLoaders
tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

The goal of this assignment is to start with the
[torch_dataset.py](https://github.com/ufal/npfl138/tree/master/labs/04/torch_dataset.py)
template and implement a simple image augmentation preprocessing.

#### Tests Start: torch_dataset_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 torch_dataset.py --epochs=1 --batch_size=100`
```
accuracy: 0.1297 - loss: 2.2519 - val_accuracy: 0.2710 - val_loss: 1.9796
```

2. `python3 torch_dataset.py --epochs=1 --batch_size=50 --augment`
```
accuracy: 0.1354 - loss: 2.2565 - val_accuracy: 0.2690 - val_loss: 1.9889
```
#### Tests End:
