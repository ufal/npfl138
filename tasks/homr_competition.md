### Assignment: homr_competition
#### Date: Deadline: May 20, 22:00
#### Points: 3 points+5 bonus

Tackle the **h**andwritten **o**ptical **m**usic **r**ecognition in this
assignment. The inputs are grayscale images of monophonic scores starting with
a clef, key signature, and a time signature, followed by several staves. The
dataset ([demo here](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/demos/homr_train.html))
is loadable using the [HOMRDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/homr_dataset/)
class, and is downloaded automatically if missing (note that it has ~500MB, so
it might take a while). No other data or pretrained models are allowed for
training.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions).
The evaluation is performed using the same metric as in `speech_recognition`, by
computing edit distance to the gold sequence, normalized by its length (the
`HOMRDataset.EditDistanceMetric` is again provided by the
[HOMRDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/homr_dataset/).
Everyone who submits a solution with at most
_3%_ test set edit distance gets 3 points; the remaining 5 bonus points are
distributed depending on the relative ordering of your solutions.
You can evaluate the predictions as usual using the
[HOMRDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/homr_dataset/)
module, either by running `python3 -m npfl138.datasets.homr_dataset --evaluate=path --dataset=dev/test`
or by calling the [HOMRDataset.evaluate](http://localhost:1234/datasets/homr_dataset/#npfl138.datasets.homr_dataset.HOMRDataset.evaluate)
method.

Start with the
[homr_competition.py](https://github.com/ufal/npfl138/tree/master/labs/12/homr_competition.py)
template, which loads the data and generates test set annotations in the required format.
