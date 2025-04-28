### Assignment: tagger_ner
#### Date: Deadline: ~~Apr 30~~ May 07, 22:00
#### Points: 2 points
#### Tests: tagger_ner_tests
#### Examples: tagger_ner_examples

This assignment is an extension of `tagger_we` task. Using the
[tagger_ner.py](https://github.com/ufal/npfl138/tree/master/labs/08/tagger_ner.py)
template, implement optimal decoding of named entity spans from
BIO-encoded tags. In a valid sequence, the tags are `O`, `B-TYPE`, `I-TYPE`, and
the `I-TYPE` tag must follow either `B-TYPE` or `I-TYPE` tags.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is a true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

To make debugging easier, the first test below includes a link to tag sequences
predicted on the development set using the optimal decoding; you can print the
tag sequences your solution predicts using the `--show_predictions` option.

**Your implementation of `constrained_decoding` must be fast enough because
during ReCodEx evaluation it is called 30 times on every batch.**

#### Tests Start: tagger_ner_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_ner.py --epochs=2 --max_sentences=10 --seed=45`
```
Epoch 1/2 0.0s train_loss=2.1919 train_accuracy=0.1314 dev_loss=2.1578 dev_accuracy=0.0566 dev_f1_constrained=0.0552 dev_f1_greedy=0.0420
Epoch 2/2 0.0s train_loss=2.1201 train_accuracy=0.9086 dev_loss=2.0992 dev_accuracy=0.4292 dev_f1_constrained=0.0435 dev_f1_greedy=0.0215
```
[The optimally decoded tag sequences on the development set](//ufal.mff.cuni.cz/~straka/courses/npfl138/2425/tasks/figures/tagger_ner.test-1.txt)

2. `python3 tagger_ner.py --epochs=2 --max_sentences=2000 --batch_size=25 --label_smoothing=0.1 --seed=45`
```
Epoch 1/2 4.4s train_loss=1.5484 train_accuracy=0.7966 dev_loss=1.2449 dev_accuracy=0.8227 dev_f1_constrained=0.0000 dev_f1_greedy=0.0000
Epoch 2/2 4.6s train_loss=1.1883 train_accuracy=0.8105 dev_loss=1.1551 dev_accuracy=0.8238 dev_f1_constrained=0.0211 dev_f1_greedy=0.0182
```
#### Tests End:
#### Examples Start: tagger_ner_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_ner.py --epochs=5 --seed=45`
```
Epoch 1/5 32.5s train_loss=0.7204 train_accuracy=0.8358 dev_loss=0.5612 dev_accuracy=0.8450 dev_f1_constrained=0.2035 dev_f1_greedy=0.1596
Epoch 2/5 39.4s train_loss=0.3630 train_accuracy=0.8907 dev_loss=0.4482 dev_accuracy=0.8784 dev_f1_constrained=0.4592 dev_f1_greedy=0.4040
Epoch 3/5 38.5s train_loss=0.1834 train_accuracy=0.9469 dev_loss=0.4306 dev_accuracy=0.8913 dev_f1_constrained=0.4959 dev_f1_greedy=0.4563
Epoch 4/5 46.4s train_loss=0.0904 train_accuracy=0.9743 dev_loss=0.4398 dev_accuracy=0.8877 dev_f1_constrained=0.4983 dev_f1_greedy=0.4499
Epoch 5/5 44.1s train_loss=0.0505 train_accuracy=0.9857 dev_loss=0.4580 dev_accuracy=0.8917 dev_f1_constrained=0.5049 dev_f1_greedy=0.4601
```

- `python3 tagger_ner.py --epochs=5 --batch_size=25 --label_smoothing=0.1 --seed=45`
```
Epoch 1/5 24.4s train_loss=1.2252 train_accuracy=0.8285 dev_loss=1.0832 dev_accuracy=0.8320 dev_f1_constrained=0.1185 dev_f1_greedy=0.0970
Epoch 2/5 20.2s train_loss=0.9486 train_accuracy=0.8621 dev_loss=0.9791 dev_accuracy=0.8667 dev_f1_constrained=0.3262 dev_f1_greedy=0.3106
Epoch 3/5 20.0s train_loss=0.8015 train_accuracy=0.9226 dev_loss=0.9255 dev_accuracy=0.8851 dev_f1_constrained=0.4554 dev_f1_greedy=0.4225
Epoch 4/5 19.8s train_loss=0.7081 train_accuracy=0.9596 dev_loss=0.8915 dev_accuracy=0.8947 dev_f1_constrained=0.5209 dev_f1_greedy=0.4865
Epoch 5/5 21.9s train_loss=0.6566 train_accuracy=0.9776 dev_loss=0.8882 dev_accuracy=0.8962 dev_f1_constrained=0.5254 dev_f1_greedy=0.4984
```
#### Examples End:
