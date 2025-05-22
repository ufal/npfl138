### Assignment: tacotron
#### Date: Deadline: Jun 30, 22:00
#### Points: 3 points

In this assignment, you will implement a speech synthesis system based on the
Tacotron 2 architecture, generating a mel spectrogram given an input text. To
pass in ReCodEx, minor toy dataset is used.

Later in the examination period, there will be instructions for those who would
like to create a working TTS system, describing how to train this implementation
on real data and how to utilize some existing vocoder.

The template `tacotron.py` will be available soon.

To obtain the same results as in the below tests, use _lazy_ convolutional and
_lazy_ linear layers, and create all layers in the same order as mentioned in
the comments. However, in ReCodEx, any order of layer creation should pass, as
should non-lazy layers.

