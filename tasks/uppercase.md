### Assignment: uppercase
#### Date: Deadline: Mar 19, 22:00
#### Points: 4 points+5 bonus

This assignment introduces first NLP task. Your goal is to implement a model
which is given Czech lowercased text and tries to uppercase appropriate letters.
To load the dataset, use
[uppercase_data.py](https://github.com/ufal/npfl138/tree/master/labs/03/uppercase_data.py)
module which loads (and if required also downloads) the data. While the training
and the development sets are in correct case, the test set is lowercased.

This is an _open-data task_, where you submit only the uppercased test set
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py/ipynb file**.

The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions). Everyone who submits
a solution achieving at least _98.5%_ accuracy gets 4 basic points; the
remaining 5 bonus points are distributed depending on relative ordering of your
solutions. The accuracy is computed per-character and can be evaluated
by running [uppercase_data.py](https://github.com/ufal/npfl138/tree/master/labs/03/uppercase_data.py)
with `--evaluate` argument, or using its `evaluate_file` method.

You may want to start with the
[uppercase.py](https://github.com/ufal/npfl138/tree/master/labs/03/uppercase.py)
template, which uses the
[uppercase_data.py](https://github.com/ufal/npfl138/tree/master/labs/03/uppercase_data.py)
to load the data, generate an alphabet of given size containing most frequent
characters, and generate sliding window view on the data. The template also
comments on possibilities of character representation.

**Do not use RNNs, CNNs, or Transformer** in this task (if you have doubts, contact me);
fully connected layers (and therefore also embedding layers), any activations,
residual connections, and any regularization layers are fine.
