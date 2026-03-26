### Assignment: svhn_competition
#### Date: Deadline: Apr 08, 22:00
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing the pretrained EfficientNetV2-B0 backbone
(or any other model from the [timm](https://huggingface.co/docs/timm) library).

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/svhn)
consists of photos of house numbers with annotations of all digits appearing in
every photo, including their bounding boxes. The dataset can be loaded using
the [SVHN](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/svhn) class.

Your goal is to produce annotations of all digits in every test set image, each
annotation consisting of a digit label and pixel coordinates of the bounding
box of the digit. All annotations of a single image should be stored on one
line, each as a space-separated five-tuple _label top left bottom right_. An annotation
is considered correct if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy on the individual images.
You can evaluate your predictions either by running
`python3 -m npfl138.datasets.svhn --evaluate=path --dataset=dev/test`
or using [SVHN.evaluate](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/svhn/#npfl138.datasets.svhn.SVHN.evaluate)
or [SVHN.evaluate_file](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/svhn/#npfl138.datasets.svhn.SVHN.evaluate_file)
methods. You can also visualize your predictions by using the
`python3 -m npfl138.datasets.svhn --visualize=path --dataset=dev/test` command
or the [visualize](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/svhn/#npfl138.datasets.svhn.SVHN.visualize) method.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions).
Everyone who submits a solution achieving at least _20%_ test set accuracy gets
5 points; the remaining 5 bonus points are distributed depending on the relative ordering
of your solutions. Note that I usually need at least _35%_ development set
accuracy to achieve the required test set performance.

You should start with the
[svhn_competition.py](https://github.com/ufal/npfl138/tree/master/labs/06/svhn_competition.py)
template, which generates the test set annotation in the required format.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Non-maximum suppression is available as
[torchvision.ops.nms](https://pytorch.org/vision/main/generated/torchvision.ops.nms.html) or
[torchvision.ops.batched_nms](https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html),
and if you want to use focal loss, its implementation is provided by
[torchvision.ops.sigmoid_focal_loss](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html)._
