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

In terms of value ranges, there are two types of quantization: asymmetric and symmetric.

Asymmetric quantization maps floating-point values to unsigned integer values, typically in the range of 0 to 255 (for 8-bit quantization). This type of quantization is well-suited for data with a non-zero minimum value or where the zero value has a special meaning.

Symmetric quantization maps floating-point values to signed integer values, typically in the range of -127 to 127 (for 8-bit quantization). In this case, the zero value in the original matrix or tensor remains zero after quantization


## Symmetric Quantization
The symmetric quantization process can be expressed as follows:

1. Calculate the scale factor $\alpha$

$$
\alpha = \frac{\max(|x|)}{2^{b-1} - 1}
$$

where $x$ is the input tensor, $b$ is the number of bits used for quantization, and $\max(|x|)$ is the maximum absolute value of $x$.

2. Quantize the input tensor $x$ to produce the quantized tensor $x\_q$:


$$
x\_q = \operatorname{clamp}\left(\left\lfloor \frac{x}{\alpha} \right\rfloor, -2^{b-1}, 2^{b-1} - 1\right)

$$

where $\lfloor \cdot \rfloor$ is the floor function, and $\text{clamp}(\cdot, a, b)$ is a function that clamps its input to the range $[a, b]$.

Now, let's see in code how this works.

```
import torch

original_tensor = torch.randn(10) * 200 + 50
print(f"Max value: {original_tensor.max()}")
print(f"Min value: {original_tensor.min()}")

#Set 0 as first element
original_tensor[0] = 0
```

Output:

```
Max value: 376.07928466796875
Min value: -194.05178833007812
```

Let's look at the tensor:

```
tensor([   0.0000,   -7.2955, -174.8314, -194.0518,  174.6870,  376.0793,
         -29.7248, -143.4094,  260.2398,  -57.8868])
```

Based on the formula for the Symmetric Quantization, we need a clamp function:

```
def clamp(params_q: torch.Tensor, lower_bound: float, upper_bound: float) -> torch.Tensor:
    """
    Clamp all elements in the tensor to a minimum and maximum value.

    Args:
        params_q (torch.Tensor): The input tensor.
        lower_bound (float): The minimum value.
        upper_bound (float): The maximum value.

    Returns:
        torch.Tensor: A tensor with the same shape as `params_q` where all elements are in the range [`lower_bound`, `upper_bound`].
    """
    return torch.clamp(params_q, lower_bound, upper_bound)
```

And let's write the symetric quantization which will return the quantized tensor and the scale

```
def symmetric_quantization(params: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    """
    Quantize the input tensor using symmetric quantization.

    Args:
        params (torch.Tensor): The input tensor.
        bits (int): The number of bits to use for quantization.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing the quantized tensor and the scale factor.
    """
    alpha = torch.max(torch.abs(params))
    scale = alpha / (2**(bits-1)-1)
    lower_bound = -2**(bits-1)
    upper_bound = 2**(bits-1)-1
    quantized = clamp(torch.round(params / scale), lower_bound, upper_bound).long()
    return quantized, scale
```

Now let's call the symmetric_quantization function and pass the tensor and the bits as 8:

```
symmetric_q_tensor, symmetric_scale = symmetric_quantization(original_tensor, 8)

```

This results into:

```
Symmetric scale: 2.961254119873047
tensor([  0,  -2, -59, -66,  59, 127, -10, -48,  88, -20])
```
Here the 0 remains the same as in the original tensor. 

How about going back, dequantize the tensor? Let's write a small function for it:

```
def symmetric_dequantize(params_q: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Dequantize the input tensor using symmetric dequantization.

    Args:
        params_q (torch.Tensor): The input tensor.
        scale (float): The scale factor.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    return params_q.float() * scale
```

When calling this function with the symmetric_q tensor, we get back this:

```
deq_tensor = symmetric_dequantize(symmetric_q_tensor, symmetric_scale)

```

```
 tensor([   0.0000,   -5.9225, -174.7140, -195.4428,  174.7140,  376.0793,
         -29.6125, -142.1402,  260.5904,  -59.2251])
```

Let's put this dequantized and original tensors close to each other:

```
Original: tensor([   0.0000,   -7.2955, -174.8314, -194.0518,  174.6870,  376.0793,
         -29.7248, -143.4094,  260.2398,  -57.8868])

Dequantized_symmetric:  tensor([   0.0000,   -5.9225, -174.7140, -195.4428,  174.7140,  376.0793,
         -29.6125, -142.1402,  260.5904,  -59.2251])
```

Here you can see some numbers match, some are different. Let's calculate the 

```
def quantization_error(original_tensor: torch.Tensor, deq_tensor: torch.Tensor) -> float:
    """
    Calculate the mean squared error between the original and the dequantized values.

    Args:
        original_tensor (torch.Tensor): The original tensor.
        deq_tensor (torch.Tensor): The dequantized tensor.

    Returns:
        float: The mean squared error.
    """
    return torch.mean((original_tensor - deq_tensor)**2)

```

And let's call it

```
q_error = quantization_error(original_tensor, deq_tensor)


0.7371761798858643
```

## Asymmetric Quantization

The asymmetric quantization process can be expressed as follows:

1. Calculate the scale factor $\alpha$ and the zero point $z$:

$$
\begin{align}
\alpha &= \frac{\max(x) - \min(x)}{2^b - 1} \\

z &= -\left\lfloor \frac{\min(x)}{\alpha} \right\rfloor
\end{align}
$$

where $x$ is the input tensor, $b$ is the number of bits used for quantization, $\max(x)$ is the maximum value of $x$, and $\min(x)$ is the minimum value of $x$.

2. Quantize the input tensor $x$ to produce the quantized tensor $x\_q$:

$$
x\_q = \text{clamp}\left(\left\lfloor \frac{x}{\alpha} + z \right\rfloor, 0, 2^b - 1\right)
$$

where $\lfloor \cdot \rfloor$ is the floor function, and $\text{clamp}(\cdot, a, b)$ is a function that clamps its input to the range $[a, b]$.

As for the symmetric quantization, let's see how to implement it in code

```
def asymmetric_quantization(params: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, int]:
    """
    Quantize the input tensor using asymmetric quantization.

    Args:
        params (torch.Tensor): The input tensor.
        bits (int): The number of bits to use for quantization.

    Returns:
        tuple[torch.Tensor, float, int]: A tuple containing the quantized tensor, the scale factor, and the zero point.
    """
    alpha = params.max()
    beta = params.min()
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*torch.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    quantized = clamp(torch.round(params / scale + zero), lower_bound, upper_bound).long()
    return quantized, scale, zero

```

This function will return the quantized tensor, the scale and the zero. For the clamp we use the same function as before.

Let's run the function with the same original tensor:

```
asymmetric_q_tensor, asymmetric_scale, asymmetric_zero = asymmetric_quantization(original_tensor, 8)
```

Which results into the below:

```
Asymmetric scale: 2.2358081340789795, 
Zero: 87
tensor([ 87,  84,   9,   0, 165, 255,  74,  23, 203,  61])
```
Here you can see that the 0 is not anymore 0 but 87.

As for the symmetric quantization, let's dequantized the asymmetric quantized tensor:

```
def asymmetric_dequantize(asym_q_tensor: torch.Tensor, scale: float, zero: int) -> torch.Tensor:
    """
    Dequantize the input tensor using asymmetric dequantization.

    Args:
        asym_q_tensor (torch.Tensor): The input tensor.
        scale (float): The scale factor.
        zero (int): The zero point.

    Returns:
        torch.Tensor: The dequantized tensor.
    """
    return (asym_q_tensor.float() - zero) * scale
```

And call it on the asymmetric quantized tensor and put it next to the original tensor:

```
deq_tensor = asymmetric_dequantize(asymmetric_q_tensor, asymmetric_scale, asymmetric_zero)
```

```
Original: tensor([   0.0000,   -7.2955, -174.8314, -194.0518,  174.6870,  376.0793,
         -29.7248, -143.4094,  260.2398,  -57.8868])

Dequantized_Asymmetric: tensor([   0.0000,   -6.7074, -174.3930, -194.5153,  174.3930,  375.6158,
         -29.0655, -143.0917,  259.3537,  -58.1310])

```

Also in this case some values match and some are a bit different. Let's calculate the error:

```
q_error = quantization_error(original_tensor, deq_tensor)

0.24344968795776367
```
