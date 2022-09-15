# Stable Diffusion 

It is undeniable that Diffusion Models are the hottest topic in the AI world right now. From Dall-E 2 from Open AI, Imagen from Google and Midjourney, everybody who works in this space have heard of them. 

All these models were not really open to everybody, some got an early access to try them out, some are still waiting or you even have to pay to use some of these models. 

Stability.AI released in August 22nd of this year ***Stable Diffusion*** which is an open source latent diffusion model that is able to do text to image and image to image. 
The model is available on the [Hugging Face](https://huggingface.co/CompVis/stable-diffusion) and can be run through the Hugging Face's diffusers as in [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) or also directly on HF's website [here](https://huggingface.co/spaces/stabilityai/stable-diffusion).

In this post however I will write about how to run it locally on GPU if you have it available.
Before showing how to run SD locally, a short introduction to what Diffusion Models are.

## Diffusion Models

Diffusion Models are generative models like GANs.
In the original paper from 2015 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf) it is mentioned that "The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly
destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data".
What this basically means is that:
- First we apply lot of noise (Gaussian Noise) to an image (Forward Diffusion Process)
- A Neural Network is then tasked to remove this noise (Reverse Diffusion Process)

**Forward Diffusion Process**
![Forward_Process](/images/Forward_Process.png)

The noise amount applied is not always the same at each timestep. This is regulated by a schedule.There is linear schedule (more aggressive) and cosine schedule (less aggressive) which seems to be a better choice in retaining information and avoid having uninformative data at the end.

**Reverse Diffusion Process**
![Reverse_Process](/images/Reverse_Process.png)

By doing this and when having a trained model that is capable of doing this task, we are able to start from complete noise and output a new image.

The Neural Network trained to remove the noise does this process step by step, gradually removing the noise until the picture is clear. 

The model architecture is UNET with a bottleneck in the middle. It takes an image as input which goes thorugh ResNet and Downsample blocks. After the bottleneck the image goes through upsample blocks.


