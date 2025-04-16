### Assignment: lemmatizer_competition
#### Date: Deadline: Apr 30, 22:00
#### Points: 4 points+5 bonus

In this assignment, you should extend `lemmatizer_noattn` or `lemmatizer_attn`
into a real-world Czech lemmatizer. As in `tagger_competition`, we will use
Czech PDT dataset loadable using the
[morpho_dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/morpho_dataset/)
module.

You can also use the same additional data as in the `tagger_competition`
assignment.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2425-summer#competitions). Everyone who submits
a solution with at least 97.0% exact match accuracy gets 4 points; the remaining 5 bonus points
are distributed depending on relative ordering of your solutions. Lastly,
**3 bonus points** will be given to anyone surpassing pre-neural-network
state-of-the-art of **98.76%**.

You can start with the
[lemmatizer_competition.py](https://github.com/ufal/npfl138/tree/master/labs/09/lemmatizer_competition.py)
template, which among others generates test set annotations in the required format. Note that
you can evaluate the predictions as usual using the
[morpho_dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/morpho_dataset/)
module, either by running `python3 -m npfl138.datasets.morpho_dataset --task=lemmatizer --evaluate=path --dataset=dev/test`
or by calling the `MorphoDataset.evaluate` method.
