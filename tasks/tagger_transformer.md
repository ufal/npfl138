### Assignment: tagger_transformer
#### Date: Deadline: May 07, 22:00
#### Points: 3 points
#### Tests: tagger_transformer_tests
#### Examples: tagger_transformer_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl138/tree/master/labs/10/tagger_transformer.py)
template, implement a Pre-LN Transformer encoder.

#### Tests Start: tagger_transformer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_layers=0`
```
Epoch 1/1 0.2s train_loss=2.4731 train_accuracy=0.2306 dev_loss=2.0946 dev_accuracy=0.3755
```

2. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=1`
```
Epoch 1/1 0.8s train_loss=2.1833 train_accuracy=0.3348 dev_loss=1.9706 dev_accuracy=0.3419
```

3. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4`
```
Epoch 1/1 0.9s train_loss=2.1739 train_accuracy=0.3406 dev_loss=1.9658 dev_accuracy=0.3455
```

4. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/1 0.9s train_loss=2.2749 train_accuracy=0.3127 dev_loss=1.9806 dev_accuracy=0.3611
```
#### Tests End:
#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch 1/5 3.5s train_loss=1.5338 train_accuracy=0.5334 dev_loss=0.8756 dev_accuracy=0.7211
Epoch 2/5 3.1s train_loss=0.5380 train_accuracy=0.8625 dev_loss=0.5533 dev_accuracy=0.8253
Epoch 3/5 3.0s train_loss=0.2495 train_accuracy=0.9569 dev_loss=0.4516 dev_accuracy=0.8419
Epoch 4/5 3.1s train_loss=0.1323 train_accuracy=0.9784 dev_loss=0.4188 dev_accuracy=0.8474
Epoch 5/5 3.6s train_loss=0.0789 train_accuracy=0.9849 dev_loss=0.4027 dev_accuracy=0.8480
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch 1/5 8.2s train_loss=1.1116 train_accuracy=0.6405 dev_loss=0.6202 dev_accuracy=0.7765
Epoch 2/5 7.6s train_loss=0.2422 train_accuracy=0.9208 dev_loss=0.4869 dev_accuracy=0.8166
Epoch 3/5 7.6s train_loss=0.0767 train_accuracy=0.9761 dev_loss=0.4880 dev_accuracy=0.8270
Epoch 4/5 7.7s train_loss=0.0441 train_accuracy=0.9855 dev_loss=0.5365 dev_accuracy=0.8418
Epoch 5/5 7.7s train_loss=0.0353 train_accuracy=0.9876 dev_loss=0.5419 dev_accuracy=0.8410
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch 1/5 8.1s train_loss=1.0783 train_accuracy=0.6540 dev_loss=0.6045 dev_accuracy=0.7882
Epoch 2/5 6.2s train_loss=0.1865 train_accuracy=0.9400 dev_loss=0.5526 dev_accuracy=0.8086
Epoch 3/5 6.3s train_loss=0.0632 train_accuracy=0.9795 dev_loss=0.6172 dev_accuracy=0.8175
Epoch 4/5 6.2s train_loss=0.0400 train_accuracy=0.9862 dev_loss=0.8000 dev_accuracy=0.8410
Epoch 5/5 6.4s train_loss=0.0322 train_accuracy=0.9893 dev_loss=0.7473 dev_accuracy=0.8466
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/5 9.2s train_loss=1.1677 train_accuracy=0.6217 dev_loss=0.6096 dev_accuracy=0.7763
Epoch 2/5 8.7s train_loss=0.2310 train_accuracy=0.9253 dev_loss=0.5208 dev_accuracy=0.8134
Epoch 3/5 8.4s train_loss=0.0784 train_accuracy=0.9762 dev_loss=0.5758 dev_accuracy=0.8335
Epoch 4/5 8.5s train_loss=0.0506 train_accuracy=0.9838 dev_loss=0.5275 dev_accuracy=0.8334
Epoch 5/5 8.5s train_loss=0.0423 train_accuracy=0.9858 dev_loss=0.6932 dev_accuracy=0.8212
```
#### Examples End:
