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

1. `python3 tagger_ner.py --epochs=1 --max_sentences=20 --seed=219`
```
Epoch 1/1 0.1s train_loss=2.2965 train_accuracy=0.2418 dev_loss=2.2514 dev_accuracy=0.3271 dev_f1_constrained=0.0269 dev_f1_greedy=0.0359
```
[The optimally decoded tag sequences on the development set](//ufal.mff.cuni.cz/~straka/courses/npfl138/2425/tasks/figures/tagger_ner.test-1.txt)

2. `python3 tagger_ner.py --epochs=2 --max_sentences=2000 --batch_size=25 --label_smoothing=0.1 --seed=219`
```
Epoch 1/2 4.0s train_loss=1.5507 train_accuracy=0.7961 dev_loss=1.2494 dev_accuracy=0.8227 dev_f1_constrained=0.0000 dev_f1_greedy=0.0000
Epoch 2/2 4.0s train_loss=1.1968 train_accuracy=0.8102 dev_loss=1.1583 dev_accuracy=0.8231 dev_f1_constrained=0.0103 dev_f1_greedy=0.0094
```
#### Tests End:
#### Examples Start: tagger_ner_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_ner.py --epochs=5 --seed=219`
```
Epoch 1/5 28.9s train_loss=0.7341 train_accuracy=0.8338 dev_loss=0.5738 dev_accuracy=0.8460 dev_f1_constrained=0.2028 dev_f1_greedy=0.1669
Epoch 2/5 30.8s train_loss=0.3700 train_accuracy=0.8893 dev_loss=0.4518 dev_accuracy=0.8799 dev_f1_constrained=0.4463 dev_f1_greedy=0.4033
Epoch 3/5 30.9s train_loss=0.1880 train_accuracy=0.9455 dev_loss=0.4701 dev_accuracy=0.8935 dev_f1_constrained=0.4979 dev_f1_greedy=0.4603
Epoch 4/5 41.3s train_loss=0.0896 train_accuracy=0.9746 dev_loss=0.4528 dev_accuracy=0.8903 dev_f1_constrained=0.5013 dev_f1_greedy=0.4434
Epoch 5/5 40.1s train_loss=0.0490 train_accuracy=0.9860 dev_loss=0.5289 dev_accuracy=0.8957 dev_f1_constrained=0.5158 dev_f1_greedy=0.4781
```

- `python3 tagger_ner.py --epochs=5 --batch_size=25 --label_smoothing=0.1 --seed=219`
```
Epoch 1/5 17.8s train_loss=1.2361 train_accuracy=0.8275 dev_loss=1.0821 dev_accuracy=0.8305 dev_f1_constrained=0.0891 dev_f1_greedy=0.0765
Epoch 2/5 20.5s train_loss=0.9534 train_accuracy=0.8598 dev_loss=0.9819 dev_accuracy=0.8647 dev_f1_constrained=0.3219 dev_f1_greedy=0.2783
Epoch 3/5 20.5s train_loss=0.8028 train_accuracy=0.9205 dev_loss=0.9272 dev_accuracy=0.8846 dev_f1_constrained=0.4480 dev_f1_greedy=0.4217
Epoch 4/5 18.8s train_loss=0.7077 train_accuracy=0.9600 dev_loss=0.8968 dev_accuracy=0.8940 dev_f1_constrained=0.4951 dev_f1_greedy=0.4685
Epoch 5/5 18.0s train_loss=0.6557 train_accuracy=0.9786 dev_loss=0.8930 dev_accuracy=0.8979 dev_f1_constrained=0.5314 dev_f1_greedy=0.4994
```
#### Examples End:
