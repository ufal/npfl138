### Assignment: cifar_competition
#### Date: Deadline: Mar 25, 22:00
#### Points: 4 points+5 bonus

The goal of this assignment is to devise the best possible model for CIFAR-10.
You can load the data using the
[CIFAR10](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/cifar10/)
class. Note that the test set is different than that of official CIFAR-10.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions). Everyone who submits a solution
achieving at least _70%_ test set accuracy gets 4 points; the remaining
5 bonus points are distributed depending on the relative ordering of your solutions.
You can evaluate a generated file with predictions by running
`python3 -m npfl138.datasets.cifar10 --evaluate PREDICTIONS_FILE_PATH --dataset dev/test`;
however, only `--dataset dev` produces valid accuracy since you do not have
the test set annotations.

Note that my solutions usually need to achieve around ~85% on the development
set to score 70% on the test set.

You may want to start with the
[cifar_competition.py](https://github.com/ufal/npfl138/tree/master/labs/04/cifar_competition.py)
template which generates the test set annotation in the required format.

In this assignment, you must implement and train your model from scratch, so you
**cannot use** the `torchvision.models` or the `timm` package.
