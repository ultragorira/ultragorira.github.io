# Stable Diffusion 

It is undeniable that Diffusion Models are the hottest topic in the AI world right now. From Dall-E 2 from Open AI, Imagen from Google and Midjourney, everybody who works in this space have heard of them. 

All these models were not really open to everybody, some got an early access to try them out, some are still waiting or you even have to pay to use some of these models. 

Stability.AI released in August 22nd of this year ***Stable Diffusion*** which is an open source latent diffusion model that is able to do text to image and image to image. 
The model is available on the [Hugging Face](https://huggingface.co/CompVis/stable-diffusion) and can be run through the Hugging Face's diffusers as in [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) or also directly on HF's website [here](https://huggingface.co/spaces/stabilityai/stable-diffusion).

In this post however I will write about how to run it locally on GPU if you have it available.
Before showing how to run SD locally, a short introduction to what Diffusion Models are.

## Diffusion Models

Diffusion Models are generative models like GANs. 
