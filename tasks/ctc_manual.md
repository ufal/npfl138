### Assignment: ctc_manual
#### Date: Deadline: ~~Apr 30~~ May 07, 22:00
#### Points: 3 points
#### Tests: ctc_manual_tests
#### Examples: ctc_manual_examples

This assignment is an extension of `tagger_we` task. Using the
[ctc_manual.py](https://github.com/ufal/npfl138/tree/master/labs/08/ctc_manual.py)
template, manually implement the CTC loss computation and also greedy CTC
decoding. You can use `torch.nn.CTCLoss` during development as a reference, but
it is not available during ReCodEx evaluation.

To make debugging easier, the first test below includes a link to file
containing $α_-$, $α_*$, final $α$, and losses for all `compute_loss` calls.

**Your implementation of `compute_loss` must be fast enough because
during ReCodEx evaluation it is called 30 times on every batch.**

#### Tests Start: ctc_manual_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 ctc_manual.py --epochs=1 --max_sentences=30`
```
Epoch 1/1 0.2s train_loss=26.8515 train_edit_distance=1.6522 dev_loss=16.7245 dev_edit_distance=0.6000
```
[Here](//ufal.mff.cuni.cz/~straka/courses/npfl138/2425/tasks/figures/ctc_manual.test-1.txt)
you can find for every example in every batch its:
- matrices $α_-$ and $α_*$, each row on a single line;
- scalar $α^N(M)$, the log likelihood of all extended labelings
  corresponding to the gold regular label;
- final example loss normalized by the target sequence length.

2. `python3 ctc_manual.py --epochs=1 --max_sentences=1000 --batch_size=100`
```
Epoch 1/1 2.8s train_loss=26.5719 train_edit_distance=1.2628 dev_loss=17.6995 dev_edit_distance=0.5864
```
#### Tests End:
#### Examples Start: ctc_manual_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 ctc_manual.py --epochs=5`
```
Epoch 1/5 38.7s train_loss=2.4757 train_edit_distance=0.5985 dev_loss=1.6736 dev_edit_distance=0.5688
Epoch 2/5 41.3s train_loss=1.3058 train_edit_distance=0.4890 dev_loss=1.3966 dev_edit_distance=0.4434
Epoch 3/5 48.9s train_loss=0.7655 train_edit_distance=0.3126 dev_loss=1.3873 dev_edit_distance=0.4193
Epoch 4/5 45.6s train_loss=0.4370 train_edit_distance=0.1745 dev_loss=1.6149 dev_edit_distance=0.4158
Epoch 5/5 48.5s train_loss=0.2641 train_edit_distance=0.1024 dev_loss=1.8303 dev_edit_distance=0.4081
```
#### Examples End:
