### Assignment: 3d_recognition
#### Date: Deadline: May 14, 22:00
#### Points: 3 points+4 bonus

Your goal in this assignment is to perform 3D object recognition. The input
is voxelized representation of an object, stored as a _3D grid_ of either empty
or occupied _voxels_, and your goal is to classify the object into one of
10 classes. The data is available in two resolutions, either as
[20×20×20 data](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/modelnet20.html)
or [32×32×32 data](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/modelnet32.html).
To load the dataset, use the
[modelnet](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/modelnet/)
module.

The official dataset offers only train and test sets, with the **test set having
a different distributions of labels**. Our dataset contains also a development
set, which has **nearly the same** label distribution as the test set.

If you want, it is possible to use any model from the
[timm](https://huggingface.co/docs/timm) library this assignment; however, the
only way I know how to utilize such a pre-trained model is to render the objects
to a set of 2D images and classify them instead.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2425-summer#competitions).
Everyone who submits a solution achieving at least _88%_ test set accuracy gets
3 points; the remaining 4 bonus points are distributed depending on relative
ordering of your solutions.

You can start with the
[3d_recognition.py](https://github.com/ufal/npfl138/tree/master/labs/11/3d_recognition.py)
template, which among others generates test set annotations in the required format.
