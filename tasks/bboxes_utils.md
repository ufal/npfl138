### Assignment: bboxes_utils
#### Date: Deadline: Apr 09, 22:00
#### Points: 2 points

This is a preparatory assignment for `svhn_competition`. The goal is to
implement several bounding box manipulation routines in the
[bboxes_utils.py](https://github.com/ufal/npfl138/tree/master/labs/06/bboxes_utils.py)
module. Notably, you need to implement the following methods:
- `bboxes_to_rcnn`: convert given bounding boxes to a R-CNN-like
  representation relative to the given anchors;
- `bboxes_from_rcnn`: convert R-CNN-like representations relative to
  given anchors back to bounding boxes;
- `bboxes_training`: given a list of anchors and gold objects, assign gold
  objects to anchors and generate suitable training data (the exact algorithm
  is described in the template).

The [bboxes_utils.py](https://github.com/ufal/npfl138/tree/master/labs/06/bboxes_utils.py)
contains simple unit tests, which are evaluated when executing the module,
which you can use to check the validity of your implementation. Note that
the template does not contain type annotations because Python typing system is
not flexible enough to describe the tensor shape changes.

When submitting to ReCodEx, the method `main` is executed, returning the
implemented `bboxes_to_rcnn`, `bboxes_from_rcnn` and `bboxes_training`
methods. These methods are then executed and compared to the reference
implementation.
