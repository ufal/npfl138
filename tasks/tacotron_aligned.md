### Assignment: tacotron_aligned
#### Date: Deadline: Jun 30, 22:00
#### Points: 1 points

Extend your solution to the `tacotron` assignment to include also the alignment
loss between the whole mel spectrogram and input text, implemented by using
the `torch.nn.CTCLoss`.

The template `tacotron_aligned.py` will be available soon.

To obtain the same results as in the below tests, use _lazy_ convolutional and
_lazy_ linear layers, and create all layers in the same order as mentioned in
the comments. However, in ReCodEx, any order of layer creation should pass, as
should non-lazy layers.

