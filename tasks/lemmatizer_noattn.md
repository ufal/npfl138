### Assignment: lemmatizer_noattn
#### Date: Deadline: Apr 29, 22:00
#### Points: 3 points
#### Tests: lemmatizer_noattn_tests
#### Examples: lemmatizer_noattn_examples

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we use the same dataset as in `tagger_we` loadable again
by the [MorphoDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/morpho_dataset/)
class using [Vocabulary](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/vocabulary/)
to provide mappings between strings and integers.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl138/tree/master/labs/09/lemmatizer_noattn.py)
template and implement the following:
- Embed characters of source forms and run a bidirectional GRU encoder.
- Embed characters of target lemmas.
- Implement a training time decoder which uses gold target characters as inputs.
- Implement an inference time decoder which uses previous predictions as inputs.
- The initial state of both decoders is the output state of the corresponding
  GRU encoded form.
- If requested, tie the embeddings in the decoder.

#### Tests Start: lemmatizer_noattn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 lemmatizer_noattn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=64 --rnn_dim=32`
```
Epoch 1/1 3.4s loss=2.9493 accuracy=0.0489 dev:accuracy=0.1972
```

2. `python3 lemmatizer_noattn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
Epoch 1/1 3.3s loss=2.8960 accuracy=0.0582 dev:accuracy=0.2119
```
#### Tests End:
#### Examples Start: lemmatizer_noattn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 lemmatizer_noattn.py --epochs=3 --max_sentences=5000`
```
Epoch 1/3 13.9s loss=2.2199 accuracy=0.1772 dev:accuracy=0.3221
Epoch 2/3 13.9s loss=0.9341 accuracy=0.4397 dev:accuracy=0.4890
Epoch 3/3 13.7s loss=0.5396 accuracy=0.5995 dev:accuracy=0.6037
```

- `python3 lemmatizer_noattn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch 1/3 13.7s loss=1.8783 accuracy=0.2614 dev:accuracy=0.3906
Epoch 2/3 13.6s loss=0.7635 accuracy=0.5107 dev:accuracy=0.5269
Epoch 3/3 13.6s loss=0.4795 accuracy=0.6406 dev:accuracy=0.6186
```
#### Examples End:
