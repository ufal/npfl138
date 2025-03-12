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
The template also shows you how to use the
[npfl138.TransformedDataset](https://github.com/ufal/npfl138/tree/master/labs/npfl138/transformed_dataset.py)
module.

#### Tests Start: torch_dataset_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 torch_dataset.py --epochs=1 --batch_size=100`
```
Epoch 1/1 1.0s train_loss=2.1280 train_accuracy=0.1872 dev_loss=1.9093 dev_accuracy=0.2800
```

2. `python3 torch_dataset.py --epochs=1 --batch_size=50 --augment`
```
Epoch 1/1 1.6s train_loss=2.1056 train_accuracy=0.2052 dev_loss=1.9050 dev_accuracy=0.2970
```
#### Tests End:
