### Assignment: cags_classification
#### Date: Deadline: Apr 02, 22:00
#### Points: 4 points+5 bonus

The goal of this assignment is to use a pretrained model, for example the
EfficientNetV2-B0, to achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/cags_train.html) consists
of images of **ca**ts and do**gs** of size $224×224$, each classified in one of
the 34 breeds and each containing a mask indicating the presence of the animal.
To load the dataset, use the [npfl138.datasets.cags](https://github.com/ufal/npfl138/blob/master/labs/npfl138/datasets/cags.py)
module.

The template includes the code loading the EfficientNetV2-B0 model using
the [timm](https://huggingface.co/docs/timm) library, which automatically
downloads the model weights. However, you can use **any** model from the `timm`
library in this assignment.

An example performing classification of given images is available in
[image_classification.py](https://github.com/ufal/npfl138/tree/master/labs/05/image_classification.py).

_A note on finetuning: you should start by training only the newly added
classifier. To that end pass only the classifier parameters to the optimizer
you want to use. If you want to finetune the whole model later, you should
create another optimizer and pass it to the `TrainableModule` using another
`configure` call._

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2425-summer#competitions). Everyone who submits a solution
achieving at least _93%_ test set accuracy gets 4 points; the remaining
5 bonus points are distributed depending on relative ordering of your solutions.

You may want to start with the
[cags_classification.py](https://github.com/ufal/npfl138/tree/master/labs/05/cags_classification.py)
template which generates the test set annotation in the required format.
