### Assignment: cifar_competition
#### Date: Deadline: Mar 26, 22:00
#### Points: 4 points+5 bonus

The goal of this assignment is to devise the best possible model for CIFAR-10.
You can load the data using the
[cifar10.py](https://github.com/ufal/npfl138/tree/master/labs/04/cifar10.py)
module. Note that the test set is different than that of official CIFAR-10.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions). Everyone who submits a solution
which achieves at least _70%_ test set accuracy will get 4 points; the rest
5 points will be distributed depending on relative ordering of your solutions.
Note that my solutions usually need to achieve around ~85% on the development
set to score 70% on the test set.

You may want to start with the
[cifar_competition.py](https://github.com/ufal/npfl138/tree/master/labs/04/cifar_competition.py)
template which generates the test set annotation in the required format.

Note that in this assignment, you **cannot use** the `keras.applications` module.
