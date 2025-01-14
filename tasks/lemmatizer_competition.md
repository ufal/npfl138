### Assignment: lemmatizer_competition
#### Date: Deadline: May 7, 22:00
#### Points: 4 points+5 bonus

In this assignment, you should extend `lemmatizer_noattn` or `lemmatizer_attn`
into a real-world Czech lemmatizer. As in `tagger_competition`, we will use
Czech PDT dataset loadable using the [morpho_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/10/morpho_dataset.py)
module.

You can also use the same additional data as in the `tagger_competition`
assignment.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2324-summer#competitions). Everyone who submits
a solution with at least 96.5% label accuracy gets 4 points; the remaining 5 bonus points
are distributed depending on relative ordering of your solutions. Lastly,
**3 bonus points** will be given to anyone surpassing pre-neural-network
state-of-the-art of **98.76%**.

You can start with the
[lemmatizer_competition.py](https://github.com/ufal/npfl138/tree/past-2324/labs/10/lemmatizer_competition.py)
template, which among others generates test set annotations in the required format. Note that
you can evaluate the predictions as usual using the [morpho_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/10/morpho_dataset.py)
module, either by running with `--task=lemmatizer --evaluate=path` arguments, or using its
`evaluate_file` method.
