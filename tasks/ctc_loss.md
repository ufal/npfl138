### Assignment: ctc_loss
#### Date: Deadline: ~~Apr 30~~ ~~May 7~~ May 14, 22:00
#### Points: 2 points
#### Tests: ctc_loss_tests
#### Examples: ctc_loss_examples

This assignment is an extension of `tagger_we` task. Using the
[ctc_loss.py](https://github.com/ufal/npfl138/tree/master/labs/09/ctc_loss.py)
template, manually implement the CTC loss computation and also greedy CTC
decoding. You can use `torch.nn.CTCLoss` during development as a reference, but
it is not available during ReCodEx evaluation.

#### Tests Start: ctc_loss_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 ctc_loss.py --epochs=1 --max_sentences=30`
```
Epoch=1/1 0.4s loss=27.2595 edit_distance=2.3694 dev_loss=17.1484 dev_edit_distance=0.6000
```

2. `python3 ctc_loss.py --epochs=1 --max_sentences=1000`
```
Epoch=1/1 8.0s loss=6.5798 edit_distance=0.6902 dev_loss=2.3089 dev_edit_distance=0.5864
```
#### Tests End:
#### Examples Start: ctc_loss_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 ctc_loss.py --epochs=5`
```
Epoch=1/5 67.0s loss=2.4850 edit_distance=0.6043 dev_loss=1.6261 dev_edit_distance=0.5635
Epoch=2/5 67.6s loss=1.2934 edit_distance=0.4832 dev_loss=1.3653 dev_edit_distance=0.4375
Epoch=3/5 68.0s loss=0.7368 edit_distance=0.3033 dev_loss=1.2962 dev_edit_distance=0.3980
Epoch=4/5 68.4s loss=0.4250 edit_distance=0.1754 dev_loss=1.5679 dev_edit_distance=0.3999
Epoch=5/5 68.4s loss=0.2656 edit_distance=0.1082 dev_loss=1.7975 dev_edit_distance=0.4054
```
#### Examples End:
