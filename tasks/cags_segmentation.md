### Assignment: cags_segmentation
#### Date: Deadline: Apr 01, 22:00
#### Points: 4 points+5 bonus

The goal of this assignment is to use a pretrained model, for example the
EfficientNetV2-B0, to achieve the best image segmentation IoU score on the CAGS
dataset. The dataset and the EfficientNetV2-B0 is described in the
`cags_classification` assignment. Nevertheless, you can again use **any** model
from `timm` library in this assignment.

A mask is evaluated using the _intersection over union_ (IoU) metric, which is
the intersection of the gold and predicted masks divided by their union, and the
whole test set score is the average of its masks' IoU. The metric implementation
is provided by the
[CAGS.MaskIoUMetric](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/cags/#npfl138.datasets.cags.CAGS.MaskIoUMetric)
class of the [CAGS](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/cags/)
dataset, which can also evaluate your predictions (either by running
`python3 -m npfl138.datasets.cags --evaluate_segmentation=path --dataset=dev/test`
or using its [evaluate_segmentation_file](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/cags/#npfl138.datasets.cags.CAGS.evaluate_segmentation_file) method) as well as visuzalize them (using
`python3 -m npfl138.datasets.cags --visualize_segmentation=path --dataset=dev/test`
or the [visualize](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/cags/#npfl138.datasets.cags.CAGS.visualize) method).

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions).
Everyone who submits a solution achieving at least _87%_ test set IoU gets
4 points; the remaining 5 bonus points are distributed depending on relative
ordering of your solutions.

You may want to start with the
[cags_segmentation.py](https://github.com/ufal/npfl138/tree/master/labs/05/cags_segmentation.py)
template, which generates the test set annotation in the required format—each
mask should be encoded on a single line as a space separated sequence of
integers indicating the length of alternating runs of zeros and ones.
