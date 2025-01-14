### Assignment: lemmatizer_noattn
#### Date: Deadline: May 7, 22:00
#### Points: 3 points
#### Tests: lemmatizer_noattn_tests
#### Examples: lemmatizer_noattn_examples

The goal of this assignment is to create a simple lemmatizer. For training
and evaluation, we use the same dataset as in `tagger_we` loadable by the
updated [morpho_dataset.py](https://github.com/ufal/npfl138/tree/past-2324/labs/10/morpho_dataset.py)
module.

Your goal is to modify the
[lemmatizer_noattn.py](https://github.com/ufal/npfl138/tree/past-2324/labs/10/lemmatizer_noattn.py)
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

1. `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32`
```
Epoch=1/1 4.7s loss=3.0619 accuracy=0.0114 dev_accuracy=0.1207
```

2. `python3 lemmatizer_noattn.py --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings`
```
Epoch=1/1 5.0s loss=2.9198 accuracy=0.0515 dev_accuracy=0.1491
```
#### Tests End:
#### Examples Start: lemmatizer_noattn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 lemmatizer_noattn.py --epochs=3 --max_sentences=5000`
```
Epoch=1/3 21.4s loss=2.2488 accuracy=0.1792 dev_accuracy=0.3247
Epoch=2/3 21.2s loss=0.9973 accuracy=0.4235 dev_accuracy=0.4670
Epoch=3/3 20.8s loss=0.5733 accuracy=0.5820 dev_accuracy=0.5983
```

- `python3 lemmatizer_noattn.py --epochs=3 --max_sentences=5000 --tie_embeddings`
```
Epoch=1/3 21.1s loss=1.9168 accuracy=0.2528 dev_accuracy=0.3765
Epoch=2/3 20.6s loss=0.8213 accuracy=0.4883 dev_accuracy=0.5110
Epoch=3/3 21.0s loss=0.5173 accuracy=0.6207 dev_accuracy=0.6094
```
#### Examples End:
