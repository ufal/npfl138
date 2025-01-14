### Assignment: tagger_transformer
#### Date: Deadline: May 14, 22:00
#### Points: 3 points
#### Tests: tagger_transformer_tests
#### Examples: tagger_transformer_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl138/tree/past-2324/labs/11/tagger_transformer.py)
template, implement a Pre-LN Transformer encoder.

#### Tests Start: tagger_transformer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_layers=0`
```
Epoch=1/1 0.4s loss=2.3716 accuracy=0.2574 dev_loss=2.0588 dev_accuracy=0.3770
```

2. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=1`
```
Epoch=1/1 1.5s loss=2.2448 accuracy=0.3251 dev_loss=1.9941 dev_accuracy=0.4101
```

3. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4`
```
Epoch=1/1 1.8s loss=2.2450 accuracy=0.3248 dev_loss=2.0000 dev_accuracy=0.4027
```

4. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch=1/1 1.8s loss=2.3592 accuracy=0.2914 dev_loss=2.0048 dev_accuracy=0.3552
```
#### Tests End:
#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch=1/5 6.1s loss=1.5235 accuracy=0.5393 dev_loss=0.8757 dev_accuracy=0.7251
Epoch=2/5 5.2s loss=0.5326 accuracy=0.8594 dev_loss=0.5576 dev_accuracy=0.8222
Epoch=3/5 5.2s loss=0.2473 accuracy=0.9555 dev_loss=0.4539 dev_accuracy=0.8386
Epoch=4/5 5.2s loss=0.1297 accuracy=0.9758 dev_loss=0.4230 dev_accuracy=0.8469
Epoch=5/5 5.2s loss=0.0792 accuracy=0.9850 dev_loss=0.4167 dev_accuracy=0.8486
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch=1/5 10.8s loss=1.0994 accuracy=0.6471 dev_loss=0.5889 dev_accuracy=0.7890
Epoch=2/5 11.1s loss=0.2447 accuracy=0.9232 dev_loss=0.5102 dev_accuracy=0.8305
Epoch=3/5 12.1s loss=0.0811 accuracy=0.9757 dev_loss=0.7861 dev_accuracy=0.8317
Epoch=4/5 11.8s loss=0.0461 accuracy=0.9849 dev_loss=0.5931 dev_accuracy=0.8409
Epoch=5/5 10.4s loss=0.0314 accuracy=0.9898 dev_loss=0.9218 dev_accuracy=0.8393
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch=1/5 12.4s loss=1.0558 accuracy=0.6616 dev_loss=0.5531 dev_accuracy=0.8054
Epoch=2/5 11.3s loss=0.1999 accuracy=0.9378 dev_loss=0.4812 dev_accuracy=0.8396
Epoch=3/5 11.1s loss=0.0699 accuracy=0.9777 dev_loss=0.6371 dev_accuracy=0.8479
Epoch=4/5 10.9s loss=0.0433 accuracy=0.9857 dev_loss=0.6803 dev_accuracy=0.8456
Epoch=5/5 11.1s loss=0.0345 accuracy=0.9877 dev_loss=0.8307 dev_accuracy=0.8424
```

- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch=1/5 12.2s loss=1.1487 accuracy=0.6313 dev_loss=0.6132 dev_accuracy=0.7883
Epoch=2/5 11.2s loss=0.2610 accuracy=0.9193 dev_loss=0.5611 dev_accuracy=0.8238
Epoch=3/5 11.2s loss=0.1091 accuracy=0.9673 dev_loss=0.4797 dev_accuracy=0.8391
Epoch=4/5 11.6s loss=0.0648 accuracy=0.9795 dev_loss=0.5924 dev_accuracy=0.8328
Epoch=5/5 11.5s loss=0.0465 accuracy=0.9844 dev_loss=0.4731 dev_accuracy=0.8446
```
#### Examples End:
