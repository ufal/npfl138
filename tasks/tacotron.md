### Assignment: tacotron
#### Date: Deadline: Jun 30, 22:00
#### Points: 3 points
#### Tests: tacotron_tests

**The template has been updated on May 26 17:00, you must redownload it.**

In this assignment, you will implement a speech synthesis system based on the
Tacotron 2 architecture, generating a mel spectrogram given an input text. To
pass in ReCodEx, minor toy dataset is used.

Later in the examination period, there will be instructions for those who would
like to create a working TTS system, describing how to train this implementation
on real data and how to utilize some existing vocoder.

Start with the [tacotron.py](https://github.com/ufal/npfl138/tree/master/labs/14/tacotron.py)
template, which contains extensive comments indicating how the architecture
should look like and how the training should be performed.

The way the locally-executed tests are performed changed in this assignment
(the `--recodex` option is now being passed in them). This way, you can mix
lazy and non-lazy layers arbitrarily and create layers in any order.
#### Tests Start: tacotron_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tacotron.py --recodex --epochs=2 --batch_size=4 --encoder_layers=1 --encoder_dim=32 --prenet_dim=16 --prenet_layers=1 --postnet_dim=16 --postnet_layers=2 --attention_dim=24 --attention_rnn_dim=16 --decoder_dim=20`
```
The torch.std of the first batch returned by Encoder: 0.5400
The torch.std of the first batch returned by Attention: 0.5991
The torch.std of the first batch returned by Decoder: (0.0580, 0.0019)
The torch.std of the first batch returned by Postnet: 1.0325
The first batch loss values: (mse=5.9784, bce=0.5616)
Epoch 1/2 3.9s train_loss=6.6053
Epoch 2/2 3.6s train_loss=6.1535
```

2. `python3 tacotron.py --recodex --epochs=1 --batch_size=2 --encoder_layers=2 --encoder_dim=16 --prenet_dim=8 --prenet_layers=2 --postnet_dim=14 --postnet_layers=1 --attention_dim=20 --attention_rnn_dim=10 --decoder_dim=24 --dropout=0.1`
```
The torch.std of the first batch returned by Encoder: 0.5572
The torch.std of the first batch returned by Attention: 0.5442
The torch.std of the first batch returned by Decoder: (0.0524, 0.0028)
The torch.std of the first batch returned by Postnet: 1.0133
The first batch loss values: (mse=6.3396, bce=0.6761)
Epoch 1/1 6.3s train_loss=6.6926
```
#### Tests End:
