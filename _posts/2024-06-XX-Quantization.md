# Quantization 

In this blog post I will be talking about a topic that's pretty relevant for nowaday's AI scenario. When talking about model size, we refer to the number of parameters, e.g. the current biggest Llama 3 has 70 billion parameter, and even bigger one (still training at the time of this blog post) will have 400 billion parameter. With such size it is not possible to run the model on basic consumer-grade hardware. This is because when doing inference, all the parameters are loaded into memory. Here is where Quantization comes in handy.  


## What is Quantization?

Quantization is the process of reducing the size of a model. In practice this means reducing the amount of bits needed to represent each parameter. The most commmon apporach is to convert floating-point to integers. 
With quantization there are also other positive aspects such as increase speed. For example, it is faster to multiply integer than floating point numbers.
One downside of quantization is the loss of precision.

## Data Types


