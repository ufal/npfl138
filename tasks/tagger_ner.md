### Assignment: tagger_ner
#### Date: Deadline: Apr 30, 22:00
#### Points: 2 points
#### Tests: tagger_ner_tests
#### Examples: tagger_ner_examples

This assignment is an extension of `tagger_we` task. Using the
[tagger_ner.py](https://github.com/ufal/npfl138/tree/master/labs/09/tagger_ner.py)
template, implement optimal decoding of named entity spans from
BIO-encoded tags. In a valid sequence, the `I-TYPE` tag must follow either
`B-TYPE` or `I-TYPE` tags.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

To make debugging easier, the first test below includes a link to tag sequences
predicted on the development set using the optimal decoding.

#### Tests Start: tagger_ner_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_ner.py --epochs=2 --max_sentences=10 --seed=52`
```
Epoch=1/2 0.1s loss=2.2081 accuracy=0.0286 dev_loss=2.1559 dev_accuracy=0.3208 dev_f1_constrained=0.0268 dev_f1_greedy=0.0292
Epoch=2/2 0.1s loss=2.1068 accuracy=0.9029 dev_loss=2.0630 dev_accuracy=0.7264 dev_f1_constrained=0.0364 dev_f1_greedy=0.0392
```
[The optimally decoded tag sequences on the development set](//ufal.mff.cuni.cz/~straka/courses/npfl138/2324/tasks/figures/tagger_ner.test-1.txt)

2. `python3 tagger_ner.py --epochs=2 --max_sentences=2000`
```
Epoch=1/2 10.3s loss=1.0373 accuracy=0.8046 dev_loss=0.7787 dev_accuracy=0.8225 dev_f1_constrained=0.0067 dev_f1_greedy=0.0057
Epoch=2/2 9.3s loss=0.6481 accuracy=0.8179 dev_loss=0.6709 dev_accuracy=0.8331 dev_f1_constrained=0.0910 dev_f1_greedy=0.0834
```

3. `python3 tagger_ner.py --epochs=2 --max_sentences=2000 --label_smoothing=0.3 --seed=44`
```
Epoch=1/2 10.2s loss=1.8852 accuracy=0.8049 dev_loss=1.7517 dev_accuracy=0.8229 dev_f1_constrained=0.0030 dev_f1_greedy=0.0029
Epoch=2/2 9.3s loss=1.7039 accuracy=0.8162 dev_loss=1.7273 dev_accuracy=0.8329 dev_f1_constrained=0.0710 dev_f1_greedy=0.0562
```
#### Tests End:
#### Examples Start: tagger_ner_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_ner.py --epochs=5`
```
Epoch=1/5 47.6s loss=0.7167 accuracy=0.8339 dev_loss=0.5699 dev_accuracy=0.8422 dev_f1_constrained=0.1915 dev_f1_greedy=0.1468
Epoch=2/5 48.0s loss=0.3642 accuracy=0.8915 dev_loss=0.4467 dev_accuracy=0.8785 dev_f1_constrained=0.4204 dev_f1_greedy=0.3684
Epoch=3/5 48.3s loss=0.1773 accuracy=0.9476 dev_loss=0.4161 dev_accuracy=0.8847 dev_f1_constrained=0.4826 dev_f1_greedy=0.4307
Epoch=4/5 49.4s loss=0.0852 accuracy=0.9755 dev_loss=0.4318 dev_accuracy=0.8877 dev_f1_constrained=0.4878 dev_f1_greedy=0.4427
Epoch=5/5 49.0s loss=0.0490 accuracy=0.9860 dev_loss=0.4354 dev_accuracy=0.8948 dev_f1_constrained=0.5214 dev_f1_greedy=0.4909
```

- `python3 tagger_ner.py --epochs=5 --label_smoothing=0.3 --seed=44`
```
Epoch=1/5 48.6s loss=1.7328 accuracy=0.8357 dev_loss=1.6601 dev_accuracy=0.8523 dev_f1_constrained=0.2548 dev_f1_greedy=0.2299
Epoch=2/5 49.5s loss=1.5568 accuracy=0.9017 dev_loss=1.6025 dev_accuracy=0.8890 dev_f1_constrained=0.4885 dev_f1_greedy=0.4546
Epoch=3/5 50.4s loss=1.4650 accuracy=0.9605 dev_loss=1.5766 dev_accuracy=0.8989 dev_f1_constrained=0.5366 dev_f1_greedy=0.5149
Epoch=4/5 50.4s loss=1.4272 accuracy=0.9806 dev_loss=1.5724 dev_accuracy=0.9011 dev_f1_constrained=0.5513 dev_f1_greedy=0.5249
Epoch=5/5 50.0s loss=1.4109 accuracy=0.9894 dev_loss=1.5728 dev_accuracy=0.9026 dev_f1_constrained=0.5533 dev_f1_greedy=0.5274
```
#### Examples End:
