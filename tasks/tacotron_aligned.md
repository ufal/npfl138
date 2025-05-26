### Assignment: tacotron_aligned
#### Date: Deadline: Jun 30, 22:00
#### Points: 1 points
#### Tests: tacotron_aligned_tests

**The template has been updated on May 26 17:00, you must redownload it.**

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
The torch.std of the first batch returned by Encoder: 0.5400
The torch.std of the first batch returned by Attention: 0.5991
The torch.std of the first batch returned by Decoder: (0.0580, 0.0019)
The torch.std of the first batch returned by Postnet: 1.0325
The first batch loss values: (mse=5.9784, bce=0.5616, ctc=21.8865)
Epoch 1/2 4.5s train_loss=26.5157
Epoch 2/2 4.2s train_loss=20.3316
```

2. `python3 tacotron_aligned.py --recodex --epochs=1 --batch_size=2 --encoder_layers=2 --encoder_dim=16 --prenet_dim=8 --prenet_layers=2 --postnet_dim=14 --postnet_layers=1 --attention_dim=20 --attention_rnn_dim=10 --decoder_dim=24 --dropout=0.1`
```
The torch.std of the first batch returned by Encoder: 0.5572
The torch.std of the first batch returned by Attention: 0.5442
The torch.std of the first batch returned by Decoder: (0.0524, 0.0028)
The torch.std of the first batch returned by Postnet: 1.0133
The first batch loss values: (mse=6.3396, bce=0.6761, ctc=20.2005)
Epoch 1/1 6.9s train_loss=23.2975
```
#### Tests End:
