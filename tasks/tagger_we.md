### Assignment: tagger_we
#### Date: Deadline: Apr 16, 22:00
#### Points: 3 points
#### Tests: tagger_we_tests
#### Examples: tagger_we_examples

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/morpho_dataset/)
module (down)loads the dataset and uses
[Vocabulary](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/vocabulary/)
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

1. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16`
```
Epoch 1/1 1.6s train_loss=2.3559 train_accuracy=0.3358 dev_loss=2.0420 dev_accuracy=0.4121
```

2. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=GRU --rnn_dim=16`
```
Epoch 1/1 1.6s train_loss=2.1929 train_accuracy=0.3318 dev_loss=1.5136 dev_accuracy=0.5596
```
#### Tests End:
#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=64`
```
Epoch 1/5 11.8s train_loss=1.0064 train_accuracy=0.6990 dev_loss=0.4203 dev_accuracy=0.8638
Epoch 2/5 11.2s train_loss=0.1147 train_accuracy=0.9717 dev_loss=0.3357 dev_accuracy=0.8799
Epoch 3/5 11.6s train_loss=0.0319 train_accuracy=0.9912 dev_loss=0.3699 dev_accuracy=0.8677
Epoch 4/5 11.4s train_loss=0.0193 train_accuracy=0.9950 dev_loss=0.3772 dev_accuracy=0.8730
Epoch 5/5 11.5s train_loss=0.0122 train_accuracy=0.9969 dev_loss=0.4070 dev_accuracy=0.8704
```

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=GRU --rnn_dim=64`
```
Epoch 1/5 12.0s train_loss=0.7531 train_accuracy=0.7726 dev_loss=0.3586 dev_accuracy=0.8909
Epoch 2/5 11.2s train_loss=0.0751 train_accuracy=0.9801 dev_loss=0.3172 dev_accuracy=0.8855
Epoch 3/5 11.1s train_loss=0.0232 train_accuracy=0.9927 dev_loss=0.3037 dev_accuracy=0.8971
Epoch 4/5 11.0s train_loss=0.0144 train_accuracy=0.9955 dev_loss=0.3446 dev_accuracy=0.8841
Epoch 5/5 11.3s train_loss=0.0088 train_accuracy=0.9974 dev_loss=0.3267 dev_accuracy=0.8940
```
#### Examples End:
