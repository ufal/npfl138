### Assignment: homr_competition
#### Date: Deadline: May 21, 22:00
#### Points: 3 points+5 bonus

Tackle the **h**andwritten **o**ptical **m**usic **r**ecognition in this
assignment. The inputs are grayscale images of monophonic scores starting with
a clef, key signature, and a time signature, followed by several staves. The
[dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/homr_train.html)
is loadable using the
[homr_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/12/homr_dataset.py)
module, and is downloaded automatically if missing (note that it has ~500MB, so
it might take a while). No other data or pretrained models are allowed for
training.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions).
The evaluation is performed using the same metric as in `speech_recognition`, by
computing edit distance to the gold sequence, normalized by its length (the
`EditDistanceMetric` is again provided by the
[homr_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/12/homr_dataset.py)).
Everyone who submits a solution with at most
_3%_ test set edit distance gets 3 points; the remaining 5 bonus points are
distributed depending on relative ordering of your solutions.
You can evaluate the predictions as usual using the
[homr_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/12/homr_dataset.py)
module, either by running with the `--evaluate=path` argument, or using its
`evaluate_file` method.

You can start with the
[homr_competition.py](https://github.com/ufal/npfl138/tree/past-2324/labs/12/homr_competition.py)
template, which among others generates test set annotations in the required format.
