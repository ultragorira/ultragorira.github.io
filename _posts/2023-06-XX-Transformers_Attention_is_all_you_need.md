# Transformers - Attention is all you need 

Neural Networks are the foundation of deep learning. You might have seen something like while reading about Deep Learning:

![MLP](/images/Transformers/MLP.PNG)

That's a Multilayer Perceptron or MLP, one of tbe most basic Neural Network architectures, where you have an input layer, a hidden layer and an output layer. 
There are several different architectures nowadays but the one that changed the game is the Transformers architecture.

# NLP and different architecture

The usage of Transformers started from in the NLP world where the data is a series of words.
Before the advent of Transformers, the preferred architectures to deal with text data were Recurrent Neural Networks (RNNs). RNNs work with any sequence data, not just text. 

![RNN_folded](/images/Transformers/RNN_Folded.PNG) ![RNN_Unfolded](/images/Transformers/Unfolded.PNG)

Even though RNNs work with sequence data, they have some limitations such as:
 - They do not work well with long sequences, and suffer of the so-called vanishing gradient. Basically when training the deep neural network, the gradients that are used to update the weights become so small or almost vanish when backpropagating to previous layers. RNNs cannot remember that much.
 - They cannot be parallelized.

An alternative to RNNs is Long short-term memory neural network which is more complex and is able to handle longer series but due to the complexity of the network, they are slow to train. 

![LSTM_Cell](/images/Transformers/LSTM_Cell.PNG)

RNN and CNN Images taken from the [Deep Learning Nanodegree](https://graduation.udacity.com/confirm/AAGWGLGC) I completed from Udacity in 2022.

Transformers on the other hand shine with sequence data. The advantages of Transformers, compared to other networks are:

- They use the Attention mechanism which we will read more about later
- They are not recurrent and they can parallelized
- They are faster to train
- Theoretically, given infinite amount of compute resources, they could handle infinite reference window (Just recently [Claude by Anthropic](https://www.anthropic.com/index/100k-context-windows) was released with a 100k tokens, roughly 75k words per time.)

## Attention mechanism

In 2017 a paper called [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) was published by Google. Back then the paper did not really catch much attention by the AI world and actually the authors of the paper would have never imagined that their work would have changed AI completely in a few years.

![Transformers](/images/Transformers/Transformers.PNG)

Transformers have an encoder and a decoder.  

