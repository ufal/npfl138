### Assignment: tagger_competition
#### Date: Deadline: Apr 15, 22:00
#### Points: 4 points+5 bonus

In this assignment, you should extend `tagger_cle`
into a real-world Czech part-of-speech tagger. We will use
Czech PDT dataset loadable using the [MorphoDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_dataset/)
class. Note that the dataset contains more than 1500 unique POS tags and that
the POS tags have a fixed structure of 15 positions (so it is possible to
generate the POS tag characters independently).

You can use the following additional data in this assignment:
- You can use outputs of a morphological analyzer loadable by the
  [MorphoAnalyzer](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_analyzer/)
  class. If a word form in train, dev, or test PDT data is known to the analyzer,
  all its _(lemma, POS tag)_ pairs are returned.
- You can use any _unannotated_ text data (Wikipedia, Czech National Corpus, …),
  and also any pre-trained word embeddings (assuming they were trained on plain
  texts).

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions).
Everyone who submits a solution with at least 93.0% label accuracy gets
4 points; the remaining 5 bonus points are distributed depending on the relative ordering
of your solutions. Lastly, **1 additional bonus point** will be given to anyone surpassing
the pre-neural-network state-of-the-art of **96.35%**.

You can start with the
[tagger_competition.py](https://github.com/ufal/npfl138/tree/master/labs/07/tagger_competition.py)
template, which among others generates test set annotations in the required format. Note that
you can evaluate the predictions as usual using the
[MorphoDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_dataset/)
class, either by running `python3 -m npfl138.datasets.morpho_dataset --evaluate=path --dataset=dev/test`
or by calling the [MorphoDataset.evaluate](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_dataset/#npfl138.datasets.morpho_dataset.MorphoDataset.evaluate)
method.
