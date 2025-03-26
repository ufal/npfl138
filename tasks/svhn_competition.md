### Assignment: svhn_competition
#### Date: Deadline: Apr 09, 22:00
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing the pretrained EfficientNetV2-B0 backbone
(or any other model from the [timm](https://huggingface.co/docs/timm) library).

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/svhn_train.html)
annotates for every photo all digits appearing on it, including their bounding
boxes. The dataset can be loaded using the [npfl138.datasets.svhn](https://github.com/ufal/npfl138/blob/master/labs/npfl138/datasets/svhn.py)
module. Similarly to the `CAGS` dataset, the `train/dev/test` are PyTorch
`torch.utils.data.Dataset`s, and every element is a dictionary with the following keys:
- `"image"`: a square 3-channel image stored using `torch.Tensor` of type `torch.uint8`,
- `"classes"`: a 1D `torch.Tensor`  with all digit labels appearing in the image,
- `"bboxes"`: a `[num_digits, 4]` 2D `torch.Tensor` with bounding boxes of every
  digit in the image, each represented as `[TOP, LEFT, BOTTOM, RIGHT]`.

Each test set image annotation consists of a sequence of space separated
five-tuples _label top left bottom right_, and the annotation is considered
correct, if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy of individual images.
You can again evaluate your predictions using the
[npfl138.datasets.svhn](https://github.com/ufal/npfl138/blob/master/labs/npfl138/datasets/svhn.py)
module, either by running with `python3 -m npfl138.datasets.svhn --evaluate=path --dataset=dev/test`
or using the `svhn.evaluate` method. Futhermore, you can visualize your
predictions by using `python3 -m npfl138.datasets.svhn --visualize=path --dataset=dev/test`.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2425-summer#competitions).
Everyone who submits a solution achieving at least _20%_ test set accuracy gets
5 points; the remaining 5 bonus points are distributed depending on relative ordering
of your solutions. Note that I usually need at least _35%_ development set
accuracy to achieve the required test set performance.

You should start with the
[svhn_competition.py](https://github.com/ufal/npfl138/tree/master/labs/06/svhn_competition.py)
template, which generates the test set annotation in the required format.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Focal loss is available as
[torchvision.ops.sigmoid_focal_loss](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html)
and non-maximum suppression as
[torchvision.ops.nms](https://pytorch.org/vision/main/generated/torchvision.ops.nms.html) or
[torchvision.ops.batched_nms](https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html)._
