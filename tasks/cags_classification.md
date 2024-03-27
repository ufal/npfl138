### Assignment: cags_classification
#### Date: Deadline: Apr 02, 22:00
#### Points: 4 points+5 bonus

The goal of this assignment is to use a pretrained model, for example the
EfficientNetV2-B0, to achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/cags_train.html) consists
of images of **ca**ts and do**gs** of size $224×224$, each classified in one of
the 34 breeds and each containing a mask indicating the presence of the animal.
To load the dataset, use the [cags_dataset.py](https://github.com/ufal/npfl138/tree/master/labs/05/cags_dataset.py)
module.

To load the EfficientNetV2-B0, use the
[keras.applications.EfficientNetV2B0](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function)
function, which constructs a Keras model, downloading the weights automatically.
However, you can use **any** model from `keras.applications` in this
assignment.

An example performing classification of given images is available in
[image_classification.py](https://github.com/ufal/npfl138/tree/master/labs/05/image_classification.py).

_A note on finetuning: each `keras.layers.Layer` has a mutable `trainable`
property indicating whether its variables should be updated – however, after
changing it, you need to call `.compile` again (or otherwise make sure the list
of trainable variables for the optimizer is updated). Furthermore, `training`
argument passed to the invocation call decides whether the layer is executed in
training regime (neurons gets dropped in dropout, batch normalization computes
estimates on the batch) or in inference regime. There is one exception though
– if `trainable == False` on a batch normalization layer, it runs in the
inference regime even when `training == True`._

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions). Everyone who submits a solution
achieving at least _93%_ test set accuracy gets 4 points; the remaining
5 bonus points are distributed depending on relative ordering of your solutions.

You may want to start with the
[cags_classification.py](https://github.com/ufal/npfl138/tree/master/labs/05/cags_classification.py)
template which generates the test set annotation in the required format.
