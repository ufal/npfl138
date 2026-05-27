### Assignment: tacotron_aligned
#### Date: Deadline: Jun 30, 22:00
#### Points: 1 points
#### Tests: tacotron_aligned_tests

Extend your solution to the `tacotron` assignment to also include the alignment
loss between the whole mel spectrogram and the input text, implemented by using
`torch.nn.CTCLoss`.

Start with the [tacotron_aligned.py](https://github.com/ufal/npfl138/tree/master/labs/14/tacotron_aligned.py)
template, which contains just one more TODO comment compared to the
`tacotron.py` template.

#### Tests Start: tacotron_aligned_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 tacotron_aligned.py --recodex --epochs=2 --batch_size=4 --encoder_layers=1 --encoder_dim=32 --prenet_dim=16 --prenet_layers=1 --postnet_dim=16 --postnet_layers=2 --attention_dim=24 --attention_rnn_dim=16 --decoder_dim=20`
```
The torch.std of the first batch returned by Encoder: 0.4903
The torch.std of the first batch returned by Attention: 0.5449
The torch.std of the first batch returned by Decoder: (0.0499, 0.0036)
The torch.std of the first batch returned by Postnet: 1.0250
The first batch loss values: (mse_post=6.1917, mse=5.2389, bce=0.5307, ctc=22.1883)
Epoch 1/2 6.2s loss=31.8102
Epoch 2/2 5.8s loss=24.3302
```

2. `python3 tacotron_aligned.py --recodex --epochs=1 --batch_size=2 --encoder_layers=2 --encoder_dim=16 --prenet_dim=8 --prenet_layers=2 --postnet_dim=14 --postnet_layers=1 --attention_dim=20 --attention_rnn_dim=10 --decoder_dim=24 --dropout=0.1`
```
The torch.std of the first batch returned by Encoder: 0.5615
The torch.std of the first batch returned by Attention: 0.5247
The torch.std of the first batch returned by Decoder: (0.0486, 0.0019)
The torch.std of the first batch returned by Postnet: 0.6518
The first batch loss values: (mse_post=5.6025, mse=5.1950, bce=0.7154, ctc=21.2525)
Epoch 1/1 9.4s loss=29.7225
```
#### Tests End:
