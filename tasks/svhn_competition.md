### Assignment: svhn_competition
#### Date: Deadline: Apr 09, 22:00
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing the pretrained EfficientNetV2-B0 backbone
(or any other model from `keras.applications`).

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/svhn_train.html)
annotates for every photo all digits appearing on it, including their bounding
boxes. The dataset can be loaded using the [svhn_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/06/svhn_dataset.py)
module. Similarly to the `CAGS` dataset, the `train/dev/test` are PyTorch
`torch.utils.data.Dataset`s, and every element is a dictionary with the following keys:
- `"image"`: a square 3-channel image stored using PyTorch tensor of type `torch.uint8`,
- `"classes"`: a 1D `np.ndarray`  with all digit labels appearing in the image,
- `"bboxes"`: a `[num_digits, 4]` 2D `np.ndarray` with bounding boxes of every
  digit in the image, each represented as `[TOP, LEFT, BOTTOM, RIGHT]`.

Each test set image annotation consists of a sequence of space separated
five-tuples _label top left bottom right_, and the annotation is considered
correct, if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy of individual images.
You can again evaluate your predictions using the
[svhn_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/06/svhn_dataset.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions).
Everyone who submits a solution achieving at least _20%_ test set accuracy gets
5 points; the remaining 5 bonus points are distributed depending on relative ordering
of your solutions. Note that I usually need at least _35%_ development set
accuracy to achieve the required test set performance.

You should start with the
[svhn_competition.py](https://github.com/ufal/npfl138/tree/past-2324/labs/06/svhn_competition.py)
template, which generates the test set annotation in the required format.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Focal loss is available
as [keras.losses.BinaryFocalCrossentropy](https://keras.io/api/losses/probabilistic_losses/#binaryfocalcrossentropy-class)
and non-maximum suppression as
[torchvision.ops.nms](https://pytorch.org/vision/main/generated/torchvision.ops.nms.html#nms) or
[torchvision.ops.batched_nms](https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html#batched-nms)._
