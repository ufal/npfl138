#### Questions@:, Lecture 1 Questions
- Considering a neural network with $D$ input neurons, a single hidden layer
  with $H$ neurons, $K$ output neurons, hidden activation $f$ and output
  activation $a$, list its parameters (including their size) and write down how
  the output is computed. [5]

- List the definitions of frequently used MLP output layer activations (the ones
  producing parameters of a Bernoulli distribution and a categorical distribution).
  Then write down three commonly used hidden layer activations (sigmoid, tanh,
  ReLU). [5]

- Formulate the Universal approximation theorem. [5]

#### Questions@:, Lecture 2 Questions
- Define maximum likelihood estimation, and show that it is equal to minimizing
  NLL, minimizing cross-entropy, and minimizing KL divergence. [10]

- Define mean squared error, show how it can be derived using MLE (define
  $p_{\textrm{model}}$, show how MLE looks using $p_{\textrm{model}}$, and prove
  that the maximum likelihood estimate is equal to minimizing MSE). [5]

- Describe gradient descent and compare it to stochastic (i.e., online) gradient
  descent and minibatch stochastic gradient descent. [5]

- Formulate conditions on the sequence of learning rates used in SGD to converge
  to optimum almost surely. [5]

- Write down the backpropagation algorithm. [5]

- Write down the mini-batch SGD algorithm with momentum. Then, formulate
  SGD with Nesterov momentum and show the difference between them. [5]

- Write down the AdaGrad algorithm and show that it tends to internally decay
  learning rate by a factor of $1/\sqrt{t}$ in step $t$. Then write down
  the RMSProp algorithm and explain how it solves the problem with the
  involuntary learning rate decay. [10]

- Write down the Adam algorithm. Then show why the bias-correction terms
  $(1-\beta^t)$ make the estimation of the first and second moment unbiased.
  [10]

#### Questions@:, Lecture 3 Questions
- Considering a neural network with $D$ input neurons, a single ReLU hidden
  layer with $H$ units and softmax output layer with $K$ units, write down the
  explicit formulas (i.e., without differential operators) of the gradient of
  all the MLP parameters (two weight matrices and two bias vectors), assuming
  input $\boldsymbol x$, target $g$ and negative log likelihood loss. [10]

- Assume a network with MSE loss generated a single output $o \in \mathbb{R}$,
  and the target output is $g$. What is the value of the loss function itself,
  and what is the explicit formula (i.e., without a differential operator) of
  the gradient of the loss function with respect to $o$? [5]

- Assume a binary-classification network with cross-entropy loss generated a single output
  $z \in \mathbb{R}$, which is passed through the sigmoid output activation
  function, producing $o = \sigma(z)$. If the target output is $g$, what is the value
  of the loss function itself, and what is the explicit formula (i.e., without
  a differential operator) of the gradient of the loss function with respect to
  $z$? [5]

- Assume a $K$-class-classification network with cross-entropy loss generated a $K$-element output
  $\boldsymbol z \in \mathbb{R}^K$, which is passed through the softmax output
  activation function, producing $\boldsymbol o=\operatorname{softmax}(\boldsymbol z)$.
  If the target distribution is $\boldsymbol g$, what is the value of the loss
  function itself, and what is the explicit formula (i.e., without a differential
  operator) of the gradient of the loss function with respect to $\boldsymbol z$? [5]

- Define $L_2$ regularization and describe its effect both on the value of the
  loss function and on the value of the loss function gradient. [5]

- Describe the dropout method and write down exactly how it is used during training and
  during inference. [5]

- Describe how label smoothing works for cross-entropy loss, both for sigmoid
  and softmax activations. [5]

- How are weights and biases initialized using the Glorot initialization? [5]

#### Questions@:, Lecture 4 Questions
- Write down the equation of how convolution of a given image is computed.
  Assume the input is an image $I$ of size $H \times W$ with $C$ channels, the
  kernel $K$ has size $N \times M$, the stride is $T \times S$, the operation
  performed is in fact cross-correlation (as usual in convolutional neural
  networks) and that $O$ output channels are computed. [5]

- Explain both `SAME` and `VALID` padding schemes and write down the output
  size of a convolutional operation with an $N \times M$ kernel on image
  of size $H \times W$ for both these padding schemes (stride is 1). [5]

- Describe batch normalization including all its parameters, and write down an
  algorithm how it is used during training and the algorithm how it is used
  during inference. Be sure to explicitly write over what is being normalized in
  case of fully connected layers and in case of convolutional layers. [10]

- Describe overall architecture of VGG-19 (you do not need to remember the exact
  number of layers/filters, but you should describe the overall order and type
  of layers that are used). [5]

#### Questions@:, Lecture 5 Questions
- Describe overall architecture of ResNet. You do not need to remember the exact
  number of layers/filters, but you should draw a bottleneck block (including
  the applications of BatchNorms and ReLUs) and state how residual connections
  work when the number of channels increases. [10]

- Draw the original ResNet block (including the exact positions of BatchNorms
  and ReLUs) and also the improved variant with full pre-activation. [5]

- Compare the bottleneck block of ResNet and ResNeXt architectures (draw the
  latter using convolutions only, i.e., do not use grouped convolutions). [5]

- Describe the CNN regularization method of networks with stochastic depth. [5]

- Compare Cutout and DropBlock. [5]

- Describe in detail how is CutMix performed. [5]

- Describe Squeeze and Excitation applied to a ResNet block. [5]

- Draw the Mobile inverted bottleneck block (including explanation of separable
  convolutions, the expansion factor, exact positions of BatchNorms and ReLUs,
  but without describing Squeeze and excitation blocks). [5]

- Assume an input image $I$ of size $H \times W$ with $C$ channels, and
  a convolutional kernel $K$ with size $N \times M$, stride $S$ and $O$ output
  channels. Write down (or derive) the equation of transposed convolution
  (or equivalently backpropagation through a convolution to its inputs). [5]

#### Questions@:, Lecture 7 Questions
- Write down how the Long Short-Term Memory (LSTM) cell operates, including
  the explicit formulas. Also mention the forget gate bias. [10]

- Write down how the Gated Recurrent Unit (GRU) operates, including
  the explicit formulas. [10]

- Why the usual dropout cannot be used on recurrent state? Describe
  how the problem can be alleviated with variational dropout. [5]

- Describe layer normalization including all its parameters, and write down how
  it is computed (be sure to explicitly state over what is being normalized in
  case of fully connected layers and convolutional layers). [5]

- Draw a tagger architecture utilizing word embeddings, recurrent
  character-level word embeddings (including how are these computed from
  individual characters), and two sentence-level bidirectional RNNs (explaining
  the bidirectionality) with a residual connection. Where would you put the
  dropout layers? [10]

#### Questions@:, Lecture 8 Questions
- In the context of named entity recognition, describe what the BIO encoding
  is and why it is used. [5]

- Write down the dynamic programming algorithm for decoding a BIO-tag sequence,
  including its asymptotic complexity. [10]

- In the context of CTC loss, describe regular and extended labelings and
  write down the algorithm for computing the log probability of a gold label
  sequence $\boldsymbol y$. [10]

- Describe how CTC predictions are performed using a beam-search. [5]

- Draw the CBOW architecture from `word2vec`, including the sizes of the inputs
  and the sizes of the outputs and used non-linearities. Also make sure to
  indicate where the embeddings are being trained. [5]

- Draw the SkipGram architecture from `word2vec`, including the sizes of the
  inputs and the sizes of the outputs and used non-linearities. Also make sure
  to indicate where the embeddings are being trained. [5]

- Describe the hierarchical softmax used in `word2vec`. [5]

- Describe the negative sampling proposed in `word2vec`, including
  the choice of distribution of negative samples. [5]

- Explain how are ELMo embeddings trained and how are they used in downstream
  applications. [5]

#### Questions@:, Lecture 9 Questions
- Considering machine translation, draw a recurrent sequence-to-sequence
  architecture without attention, both during training and during inference
  (include embedding layers, recurrent cells, classification layers,
  argmax/softmax). [5]

- Considering machine translation, draw a recurrent sequence-to-sequence
  architecture with attention, used during training (include embedding layers,
  recurrent cells, attention, classification layers).
  Then write down how exactly is the attention computed. [10]

- Explain how is word embeddings tying used in a sequence-to-sequence
  architecture, including the necessary scaling. [5]

- Write down why are subword units used in text processing, and describe the BPE
  algorithm for constructing a subword dictionary from a large corpus. [5]

- Write down why are subword units used in text processing, and describe the
  WordPieces algorithm for constructing a subword dictionary from a large
  corpus. [5]

- Pinpoint the differences between the BPE and WordPieces algorithms, both
  during dictionary construction and during inference. [5]

- Describe the Transformer encoder architecture, including the description of
  self-attention (but you do not need to describe multi-head attention), FFN
  and positions of LNs and dropouts. [10]

- Write down the formula of Transformer self-attention assuming you get
  sequence representation $\boldsymbol X \in \mathbb{R}^{n \times d}$, and then
  describe multi-head self-attention in detail, including the dimensionality
  of the individual heads. [10]

- Describe the Transformer decoder architecture, including the description of
  self-attention and masked self-attention (but you do not need to describe
  multi-head attention), FFN and positions of LNs and dropouts. Also discuss the
  difference between training and prediction regimes. [10]

#### Questions@:, Lecture 10 Questions
- Why are positional embeddings needed in Transformer architecture? Write down
  the sinusoidal positional embeddings used in the Transformer. [5]

- Compare RNN to Transformer – what are the strengths and weaknesses of these
  architectures? [5]

- Describe the BERT architecture (you do not need to describe the (multi-head)
  self-attention operation). Elaborate also on which positional embeddings
  are used and what are the GELU activations. [10]

- Describe the GELU activations and explain why are they a combination of ReLUs
  and Dropout. [5]

- Elaborate on BERT training process (what are the two objectives used and how
  exactly are the corresponding losses computed). [10]

- Describe the architecture of a Vision Transformer – how are input images
  represented, draw the Transformer encoder layer and the FFN sublayer, how is
  the distribution over predicted classes computed, what positional embeddings
  are used (and what alternative positional embeddings were tried). [10]

#### Questions@:, Lecture 11 Questions
- Define the Markov Decision Process, including the definition of the return. [5]

- Define the value function, such that all expectations are over simple random
  variables (actions, states, rewards), not trajectories. [5]

- Define the action-value function, such that all expectations are over simple
  random variables (actions, states, rewards), not trajectories. [5]

- Express the value function using the action-value function, and express the
  action-value function using the value function. [5]

- Formulate the policy gradient theorem. [5]

- Prove the part of the policy gradient theorem showing the value
  of $\nabla_{\boldsymbol\theta} v_\pi(s)$. [10]

- Assuming the policy gradient theorem, formulate the loss used by the REINFORCE
  algorithm and show how can its gradient be expressed as an expectation
  over states and actions. [5]

- Write down the REINFORCE algorithm, including the loss formula. [10]

- Show that introducing baseline does not influence validity of the policy
  gradient theorem. [5]

- Write down the REINFORCE with baseline algorithm, including both loss
  formulas. [10]

- Sketch the overall structure and training procedure of the Neural Architecture
  Search. You do not need to describe how exactly is the block produced by the
  controller. [5]

- Write down the variational lower bound (ELBO) in the form of a reconstruction
  error minus the KL divergence between the encoder and the prior (i.e., in the
  form used for model training). Then prove it is actually a lower bound on
  the log-likelihood $\log P(\boldsymbol x)$. [10]

- Draw an architecture of a variational autoencoder (VAE). Pay attention to the
  parametrization of the distribution from the encoder (including the used
  activation functions), show how to perform latent variable sampling so
  that it is differentiable with respect to the encoder parameters (the
  reparametrization trick), and write down the loss. [10]

#### Questions@:, Lecture 12 Questions
- Write down the min-max formulation of generative adversarial network (GAN)
  objective. Then describe what loss is actually used for training the generator
  in order to avoid vanishing gradients at the beginning of the training. [5]

- Write down the training algorithm of generative adversarial networks (GAN),
  including the losses minimized by the discriminator and the generator. Be sure
  to use the version of generator loss which avoids vanishing gradients at the
  beginning of the training. [10]

- Explain how the class label is used when training a conditional generative
  adversarial network (CGAN). [5]

- Illustrate that alternating SGD steps are not guaranteed to converge for
  a min-max problem. [5]

Flow Matching questions will be added later.
