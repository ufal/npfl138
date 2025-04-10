### Assignment: tensorboard_projector

You can try exploring the TensorBoard Projector with pre-trained embeddings
for 20k most frequent lemmas in
[Czech](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/cs_lemma_20k.zip)
and [English](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/demos/en_lemma_20k.zip)
– after extracting the archive, start
`tensorboard --logdir dir_where_the_archive_is_extracted`.

In order to use the Projector tab yourself, you can take inspiration from the
[projector_export.py](https://github.com/ufal/npfl138/tree/master/labs/08/projector_export.py)
script, which was used to export the above pre-trained embeddings from the
Word2vec format.
