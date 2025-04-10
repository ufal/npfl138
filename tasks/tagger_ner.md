### Assignment: tagger_ner
#### Date: Deadline: Apr 30, 22:00
#### Points: 2 points
#### Tests: tagger_ner_tests
#### Examples: tagger_ner_examples

This assignment is an extension of `tagger_we` task. Using the
[tagger_ner.py](https://github.com/ufal/npfl138/tree/master/labs/08/tagger_ner.py)
template, implement optimal decoding of named entity spans from
BIO-encoded tags. In a valid sequence, the `I-TYPE` tag must follow either
`B-TYPE` or `I-TYPE` tags.

The evaluation is performed using the provided metric computing F1 score of the
span prediction (i.e., a recognized possibly-multiword named entity is a true
positive if both the entity type and the span exactly match).

In practice, character-level embeddings (and also pre-trained word embeddings)
would be used to obtain superior results.

To make debugging easier, the first test below includes a link to tag sequences
predicted on the development set using the optimal decoding; you can print the
tag sequences your solution predicts using the `--show_predictions` option.

#### Tests Start: tagger_ner_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_ner.py --epochs=2 --max_sentences=10 --seed=45`
```
Epoch 1/2 0.0s train_loss=2.1919 train_accuracy=0.1314 dev_loss=2.1578 dev_accuracy=0.0566 dev_f1_constrained=0.0552 dev_f1_greedy=0.0420
Epoch 2/2 0.0s train_loss=2.1201 train_accuracy=0.9086 dev_loss=2.0992 dev_accuracy=0.4292 dev_f1_constrained=0.0435 dev_f1_greedy=0.0215
```
[The optimally decoded tag sequences on the development set](//ufal.mff.cuni.cz/~straka/courses/npfl138/2425/tasks/figures/tagger_ner.test-1.txt)

2. `python3 tagger_ner.py --epochs=2 --max_sentences=2000`
```
Epoch 1/2 6.1s train_loss=1.0192 train_accuracy=0.8053 dev_loss=0.7696 dev_accuracy=0.8221 dev_f1_constrained=0.0020 dev_f1_greedy=0.0038
Epoch 2/2 5.8s train_loss=0.6460 train_accuracy=0.8182 dev_loss=0.6800 dev_accuracy=0.8287 dev_f1_constrained=0.0365 dev_f1_greedy=0.0316
```

3. `python3 tagger_ner.py --epochs=2 --max_sentences=2000 --label_smoothing=0.3 --seed=44`
```
Epoch 1/2 6.4s train_loss=1.8784 train_accuracy=0.8054 dev_loss=1.7460 dev_accuracy=0.8227 dev_f1_constrained=0.0116 dev_f1_greedy=0.0106
Epoch 2/2 5.8s train_loss=1.6990 train_accuracy=0.8188 dev_loss=1.7071 dev_accuracy=0.8303 dev_f1_constrained=0.0301 dev_f1_greedy=0.0241
```
#### Tests End:
#### Examples Start: tagger_ner_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_ner.py --epochs=5`
```
Epoch 1/5 43.9s train_loss=0.7144 train_accuracy=0.8341 dev_loss=0.5723 dev_accuracy=0.8443 dev_f1_constrained=0.1848 dev_f1_greedy=0.1574
Epoch 2/5 51.2s train_loss=0.3744 train_accuracy=0.8867 dev_loss=0.4487 dev_accuracy=0.8792 dev_f1_constrained=0.4485 dev_f1_greedy=0.3906
Epoch 3/5 46.5s train_loss=0.1873 train_accuracy=0.9448 dev_loss=0.4285 dev_accuracy=0.8935 dev_f1_constrained=0.5052 dev_f1_greedy=0.4706
Epoch 4/5 46.0s train_loss=0.0857 train_accuracy=0.9756 dev_loss=0.4286 dev_accuracy=0.8923 dev_f1_constrained=0.5098 dev_f1_greedy=0.4690
Epoch 5/5 54.2s train_loss=0.0462 train_accuracy=0.9868 dev_loss=0.4729 dev_accuracy=0.8966 dev_f1_constrained=0.5348 dev_f1_greedy=0.4951
```

- `python3 tagger_ner.py --epochs=5 --label_smoothing=0.3 --seed=44`
```
Epoch 1/5 41.1s train_loss=1.7291 train_accuracy=0.8359 dev_loss=1.6504 dev_accuracy=0.8533 dev_f1_constrained=0.2422 dev_f1_greedy=0.2000
Epoch 2/5 51.7s train_loss=1.5505 train_accuracy=0.9045 dev_loss=1.5881 dev_accuracy=0.8909 dev_f1_constrained=0.4916 dev_f1_greedy=0.4724
Epoch 3/5 51.8s train_loss=1.4611 train_accuracy=0.9628 dev_loss=1.5797 dev_accuracy=0.8979 dev_f1_constrained=0.5251 dev_f1_greedy=0.4978
Epoch 4/5 63.1s train_loss=1.4246 train_accuracy=0.9823 dev_loss=1.5741 dev_accuracy=0.9011 dev_f1_constrained=0.5459 dev_f1_greedy=0.5275
Epoch 5/5 48.3s train_loss=1.4089 train_accuracy=0.9905 dev_loss=1.5733 dev_accuracy=0.9009 dev_f1_constrained=0.5431 dev_f1_greedy=0.5302
```
#### Examples End:
