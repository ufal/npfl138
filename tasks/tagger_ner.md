### Assignment: tagger_ner
#### Date: Deadline: Apr 30, 22:00
#### Points: 2 points

**The template is being finalized, final version will be released shortly.**

This assignment is an extension of `tagger_we` task. Using the
`tagger_ner.py`
template, implement optimal decoding of named entity spans from
BIO-encoded tags.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

