### Assignment: tagger_we
#### Date: Deadline: Apr 23, 22:00
#### Points: 3 points
#### Tests: tagger_we_tests
#### Examples: tagger_we_examples

In this assignment you will create a simple part-of-speech tagger. For training
and evaluation, we will use Czech dataset containing tokenized sentences, each
word annotated by gold lemma and part-of-speech tag. The
[morpho_dataset.py](https://github.com/ufal/npfl138/tree/master/labs/08/morpho_dataset.py)
module (down)loads the dataset and provides mappings between strings and integers.

Your goal is to modify the
[tagger_we.py](https://github.com/ufal/npfl138/tree/master/labs/08/tagger_we.py)
template and implement the following:
- Use specified RNN layer type (`GRU` and `LSTM`) and dimensionality.
- Create word embeddings for training vocabulary.
- Process the sentences using bidirectional RNN.
- Predict part-of-speech tags.
Note that you need to properly handle sentences of different lengths in one
batch.

#### Tests Start: tagger_we_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=LSTM --rnn_dim=16`
```
Epoch=1/1 3.1s loss=2.3541 accuracy=0.3138 dev_loss=2.0320 dev_accuracy=0.3611
```

2. `python3 tagger_we.py --epochs=1 --max_sentences=1000 --rnn=GRU --rnn_dim=16`
```
Epoch=1/1 3.2s loss=2.1970 accuracy=0.4233 dev_loss=1.5569 dev_accuracy=0.5121
```
#### Tests End:
#### Examples Start: tagger_we_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=LSTM --rnn_dim=64`
```
Epoch=1/5 21.1s loss=0.9776 accuracy=0.7080 dev_loss=0.3744 dev_accuracy=0.8814
Epoch=2/5 19.2s loss=0.1060 accuracy=0.9736 dev_loss=0.2947 dev_accuracy=0.9013
Epoch=3/5 19.4s loss=0.0291 accuracy=0.9921 dev_loss=0.2794 dev_accuracy=0.9057
Epoch=4/5 19.7s loss=0.0166 accuracy=0.9960 dev_loss=0.2976 dev_accuracy=0.9015
Epoch=5/5 19.7s loss=0.0096 accuracy=0.9978 dev_loss=0.3159 dev_accuracy=0.8957
```

- `python3 tagger_we.py --epochs=5 --max_sentences=5000 --rnn=GRU --rnn_dim=64`
```
Epoch=1/5 20.5s loss=0.7698 accuracy=0.7703 dev_loss=0.3432 dev_accuracy=0.8903
Epoch=2/5 18.9s loss=0.0735 accuracy=0.9807 dev_loss=0.2999 dev_accuracy=0.8969
Epoch=3/5 19.0s loss=0.0245 accuracy=0.9923 dev_loss=0.3244 dev_accuracy=0.8965
Epoch=4/5 19.2s loss=0.0153 accuracy=0.9955 dev_loss=0.3302 dev_accuracy=0.8929
Epoch=5/5 19.0s loss=0.0088 accuracy=0.9977 dev_loss=0.3641 dev_accuracy=0.8923
```
#### Examples End:
