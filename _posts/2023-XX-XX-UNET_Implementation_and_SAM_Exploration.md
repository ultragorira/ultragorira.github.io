# UNET Model Implementation and SAM (Segment Anything Model) exploration

![](/images/UNET/cells.jpg)

The topic of this post generated from a need of mine to find a way of removing background in specific type of images. I had not really worked much with Image Segmentation and had to investigate possible ways to achieve this in an automated way. While looking for a solution I came across [rembg](https://github.com/danielgatis/rembg) which is a Python library that removes background from an image and keeps the primary subject(s) very neatly. The library is based on U2NET which is a more recent version of UNET that came out in 2015. I wanted to learn more about this topic so in this blog post I will talk about the UNET model architecture and implementation in PyTorch. But that's not all. Last week FAIR released [SAM](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) - Segment Anything Model. This new model is absolutely mind-blowing and in this post I will also show couple of example on how to use it and off the bat results on random images. 

## UNET Architecture

![UNET](/images/UNET/u-net-architecture.PNG)

Image taken from the original paper here: https://arxiv.org/abs/1505.04597

```

```





![Original](/images/FineTuneGPT/Idonot2.PNG)

