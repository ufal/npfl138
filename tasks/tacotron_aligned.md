### Assignment: tacotron_aligned
#### Date: Deadline: Jun 30, 22:00
#### Points: 1 points
#### Tests: tacotron_aligned_tests

Extend your solution to the `tacotron` assignment to include also the alignment
loss between the whole mel spectrogram and input text, implemented by using
the `torch.nn.CTCLoss`.

Start with the [tacotron_aligned.py](https://github.com/ufal/npfl138/tree/master/labs/14/tacotron_aligned.py)
template, which contains just one more TODO comment compared to the
`tacotron.py` template.

The way the locally-executed tests are performed changed in this assignment
(the `--recodex` option is now being passed in them). This way, you can mix
lazy and non-lazy layers arbitrarily and create layers in any order.

#### Tests Start: tacotron_aligned_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tacotron_aligned.py --recodex --epochs=2 --batch_size=4 --encoder_layers=1 --encoder_dim=32 --prenet_dim=16 --prenet_layers=1 --postnet_dim=16 --postnet_layers=2 --attention_dim=24 --attention_rnn_dim=16 --decoder_dim=20`
```
Epoch 1/2 5.3s train_loss=26.6104
Epoch 2/2 5.0s train_loss=20.7103
```
To make debugging easier, here are statistics of first batches of various quantities:
```
The first batch loss values: (mse=6.5497, bce=0.5616, ctc=21.8865)
```

2. `python3 tacotron_aligned.py --recodex --epochs=2 --batch_size=2 --encoder_layers=2 --encoder_dim=16 --prenet_dim=8 --prenet_layers=2 --postnet_dim=14 --postnet_layers=1 --attention_dim=20 --attention_rnn_dim=10 --decoder_dim=24 --dropout=0.1`
```
Epoch 1/2 11.9s train_loss=22.2973
Epoch 2/2 11.9s train_loss=14.8125
```
To make debugging easier, here are statistics of first batches of various quantities:
```
The first batch loss values: (mse=5.9904, bce=0.6761, ctc=20.2005)
```
#### Tests End:
