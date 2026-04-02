### Assignment: tagger_we
#### Date: Deadline: Apr 15, 22:00
#### Points: 3 points
#### Tests: tagger_we_tests
#### Examples: tagger_we_examples

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The dataset can be loaded
by the [MorphoDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_dataset/)
class and uses [Vocabulary](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/vocabulary/)
to provide mappings between strings and integers.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl138/tree/master/labs/07/tagger_we.py)
template and implement the following:
- Use specified RNN layer type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch.

_In the alternative [tagger_we.packed.py](https://github.com/ufal/npfl138/tree/master/labs/07/tagger_we.packed.py)
template, `forward` processes a `PackedSequence` instead of a rectangular tensor
and produces also a `PackedSequence`; both templates deliver the same results,
and are both accepted by ReCodEx._

#### Tests Start: tagger_we_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_we.py --recodex --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16`
```
Epoch 1/1 2.3s loss=2.3008 accuracy=0.3416 dev:loss=1.8609 dev:accuracy=0.4439
```

2. `python3 tagger_we.py --recodex --epochs=1 --max_sentences=1000 --rnn=GRU --rnn_dim=16`
```
Epoch 1/1 2.4s loss=2.0973 accuracy=0.4885 dev:loss=1.3953 dev:accuracy=0.5740
```
#### Tests End:
#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=64`
```
Epoch 1/5 15.6s loss=1.0064 accuracy=0.6990 dev:loss=0.4203 dev:accuracy=0.8638
Epoch 2/5 14.7s loss=0.1147 accuracy=0.9717 dev:loss=0.3357 dev:accuracy=0.8799
Epoch 3/5 14.8s loss=0.0319 accuracy=0.9912 dev:loss=0.3699 dev:accuracy=0.8677
Epoch 4/5 15.1s loss=0.0193 accuracy=0.9950 dev:loss=0.3772 dev:accuracy=0.8730
Epoch 5/5 15.2s loss=0.0122 accuracy=0.9969 dev:loss=0.4070 dev:accuracy=0.8704
```

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=GRU --rnn_dim=64`
```
Epoch 1/5 15.2s loss=0.7531 accuracy=0.7726 dev:loss=0.3586 dev:accuracy=0.8909
Epoch 2/5 13.6s loss=0.0751 accuracy=0.9801 dev:loss=0.3172 dev:accuracy=0.8855
Epoch 3/5 13.9s loss=0.0232 accuracy=0.9927 dev:loss=0.3037 dev:accuracy=0.8971
Epoch 4/5 14.9s loss=0.0144 accuracy=0.9955 dev:loss=0.3446 dev:accuracy=0.8841
Epoch 5/5 14.9s loss=0.0088 accuracy=0.9974 dev:loss=0.3267 dev:accuracy=0.8940
```
#### Examples End:
