### Assignment: tagger_transformer
#### Date: Deadline: May 06, 22:00
#### Points: 3 points
#### Tests: tagger_transformer_tests
#### Examples: tagger_transformer_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl138/tree/master/labs/10/tagger_transformer.py)
template, implement a Pre-LN Transformer encoder with sinusoidal positional embeddings.

#### Tests Start: tagger_transformer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_transformer.py --recodex --epochs=1 --max_sentences=800 --transformer_layers=0`
```
Epoch 1/1 0.3s loss=2.7847 accuracy=0.1671 dev:loss=2.1306 dev:accuracy=0.3777
```

2. `python3 tagger_transformer.py --recodex --epochs=1 --max_sentences=800 --transformer_heads=1`
```
Epoch 1/1 2.1s loss=2.3620 accuracy=0.3002 dev:loss=2.0739 dev:accuracy=0.3275
```

3. `python3 tagger_transformer.py --recodex --epochs=1 --max_sentences=800 --transformer_heads=4`
```
Epoch 1/1 2.5s loss=2.3524 accuracy=0.2983 dev:loss=2.0769 dev:accuracy=0.3356
```

4. `python3 tagger_transformer.py --recodex --epochs=1 --max_sentences=800 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/1 2.6s loss=2.4369 accuracy=0.2773 dev:loss=2.1142 dev:accuracy=0.3036
```
#### Tests End:
#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch 1/5 4.3s loss=1.5539 accuracy=0.5357 dev:loss=0.8864 dev:accuracy=0.7158
Epoch 2/5 3.8s loss=0.5384 accuracy=0.8522 dev:loss=0.5587 dev:accuracy=0.8221
Epoch 3/5 3.9s loss=0.2501 accuracy=0.9565 dev:loss=0.4541 dev:accuracy=0.8391
Epoch 4/5 3.8s loss=0.1305 accuracy=0.9775 dev:loss=0.4197 dev:accuracy=0.8450
Epoch 5/5 3.8s loss=0.0788 accuracy=0.9844 dev:loss=0.4061 dev:accuracy=0.8475
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch 1/5 7.4s loss=1.0241 accuracy=0.6751 dev:loss=0.4987 dev:accuracy=0.8217
Epoch 2/5 7.0s loss=0.2107 accuracy=0.9367 dev:loss=0.5022 dev:accuracy=0.8136
Epoch 3/5 7.1s loss=0.0808 accuracy=0.9756 dev:loss=0.4734 dev:accuracy=0.8464
Epoch 4/5 7.0s loss=0.0474 accuracy=0.9845 dev:loss=0.5585 dev:accuracy=0.8467
Epoch 5/5 7.1s loss=0.0416 accuracy=0.9860 dev:loss=0.6461 dev:accuracy=0.8507
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch 1/5 7.9s loss=1.0206 accuracy=0.6722 dev:loss=0.5297 dev:accuracy=0.7991
Epoch 2/5 7.6s loss=0.1873 accuracy=0.9427 dev:loss=0.4457 dev:accuracy=0.8326
Epoch 3/5 7.5s loss=0.0712 accuracy=0.9783 dev:loss=0.4638 dev:accuracy=0.8370
Epoch 4/5 7.6s loss=0.0424 accuracy=0.9855 dev:loss=0.5281 dev:accuracy=0.8419
Epoch 5/5 7.7s loss=0.0317 accuracy=0.9888 dev:loss=0.5589 dev:accuracy=0.8400
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/5 8.5s loss=1.1099 accuracy=0.6435 dev:loss=0.5297 dev:accuracy=0.8082
Epoch 2/5 8.1s loss=0.2308 accuracy=0.9302 dev:loss=0.5058 dev:accuracy=0.8252
Epoch 3/5 8.2s loss=0.0931 accuracy=0.9721 dev:loss=0.4858 dev:accuracy=0.8254
Epoch 4/5 8.0s loss=0.0571 accuracy=0.9819 dev:loss=0.4697 dev:accuracy=0.8479
Epoch 5/5 8.1s loss=0.0447 accuracy=0.9845 dev:loss=0.5220 dev:accuracy=0.8503
```
#### Examples End:
