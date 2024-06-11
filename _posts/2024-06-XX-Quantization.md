# Quantization: A Crucial Technique in Today's AI Landscape

In this blog post, we will discuss about a topic of significant importance in the current AI scene. When discussing model size, we are referring to the number of parameters a given model possesses. For instance, the current largest Llama 3 model boasts 70 billion parameters, and an even larger one, still in the training phase at the time of this post, is expected to have around 400 billion parameters.

Such immense models pose a challenge when it comes to deployment on consumer-grade hardware, as the entirety of the parameters must be loaded into memory during the inference process. This is where the technique of Quantization proves to be indispensable. 


## What is Quantization?

Quantization is a technique used to reduce the size of a model, making it more feasible to deploy on resource-constrained hardware. In essence, quantization involves reducing the number of bits required to represent each parameter within the model.

The most common approach to quantization is to convert floating-point numbers to integers. This process not only results in a smaller model size but also has the added benefit of increased speed. For instance, multiplying integer values is computationally faster than multiplying floating-point numbers.

However, it is important to note that quantization is not without its drawbacks. One of the primary concerns is the potential loss of precision, which may negatively impact the model's performance. Therefore, striking the right balance between model size, speed, and precision is crucial when employing quantization techniques.

## Data Types

Let's begin by understanding what floating-point numbers are and how they are represented. A commonly used floating-point format is FP32, also known as single-precision floating-point. FP32 is represented using 32 bits, which are further divided into three parts:

- First bit is the sign (0 being positive)
- Next 8 bit is the exponent, the range of the number
- Last 23 bit is the fraction, precision of the number

Example from Wikipedia
![fp32](/images/Quantization/fp32.png)

There are other floating-point formats besides FP32, such as FP16 (16-bit half-precision), FP64 (64-bit double-precision), and BF16 (Brain Floating Point), which is a 16-bit format designed for deep learning applications. Each of these formats has its own trade-offs between precision, range, and memory usage.


In contrast to floating-point numbers, integers can be either signed or unsigned and are available in different bit-widths. The common bit-widths for integers are:

8-bit: This can represent signed values in the range of -128 to 127 or unsigned values from 0 to 255.
16-bit, 32-bit and 64-bit are also available. 

In PyTorch, the following integer tensor types are available:

torch.int8 (8-bit signed integers)
torch.int16 (16-bit signed integers)
torch.int32 (32-bit signed integers)
torch.int64 (64-bit signed integers)
torch.uint8 (8-bit unsigned integers)

You can look at the ranges by using torch.iinfo(torch.int8) for example => iinfo(min=0, max=255, dtype=uint8)

So why bringing up this distinction? Simply put the main purpose of bringing up the distinction between floating-point numbers and integers is to highlight the core concept of quantization. Quantization aims to reduce the number of bits required to represent the values in a model, and so to a smaller range of integer values. The end-result is a reduced memory footprint and potentially accelerating computations.
By converting floating-point numbers to integers, quantization takes advantage of the more compact and computationally efficient nature of integers. 

# Where Quantization is applied

In a simple neural network composed of stacked linear layers, each layer consists of matrices, namely weight and bias matrices. These matrices typically store their values in floating-point format to maintain high precision. This is where quantization comes into play, aiming to convert the floating-point values to integers.

The quantization process in a neural network involves the following steps for each layer:

1. Quantize the input: Convert the floating-point input values to integer values.

2. Quantize the weights and biases: Convert the floating-point weight and bias values to integer values.

3. Perform the calculation: Execute the matrix multiplications and other operations using the quantized integer values.

4. Dequantize the output: Convert the integer output values back to floating-point values, which can then be fed to the next layer.


Here's a visual example:

![quantize_dequantize](/images/Quantization/Quantize_Dequantize.PNG)

The above is just an example of how a matrix that has floating point values is quantized to a range between -128 to 127. The same is the dequantized. In the original matrix, each value takes up 4 bytes, so 32 bit. In the quantized matrix each value takes up 8 bit.
You can see that some values have slightly changed, meaning we lost some precision. so the model will not be as accurate as the not quantized one.
 