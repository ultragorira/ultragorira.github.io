# Style Transfer with VGG19

This post is an implementation of the paper [Image Style Transfer Using Convolutional Neural Networks by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). 

## What is Style Transfer?

Style transfer is the ability to apply style of one image to a content of another image. An example for applying style transfer is with the VGG19 which is a CNN with 5 pooling layers and each convolutional layer is a stack of layers. 

To obtain the target image, the one with content from one image and style from another, we pass both content and style image through the VGG19.

When passing the content image, the image will go through till the deepest conv layer, where a representation of the image will be the output. Next we pass the style image and the network will extract different features from multiple layers that represent the style of that image.
The output will be the merging of content representation and style representation.

The challenge is to get the target image which can start as a blank canvas or a copy of the input image and need to manipulate the style. For the content representation, in the paper, the output is taken from the conv layer 4_2. The output is then compared to the input image.

We need to calculate a content loss which is the mean square loss between the content representation from image and content representation of the target image.
This will measure how far apart are the two representation from each other. The aim is to minimize the loss by backpropagation. 

Similarly to comparing the content representation and target rapresentation, we do for the style where we need to compare the style representation of the style image and the target image. 
We basically check how similar features are in the same layer of the network. Similarities will be colors and textures.

By including the correlations of between multi layers of different sizes we can obtain a multi scale style representation of the style image where large and small style features are caught. 

The gram matrix defines the correlations. 
If we have a 4x4 pic passed through the conv layer with 8 of depth, the matrix will have 8 feature maps that we want to find the relationships between. 

We basically vectorize (flatten the values).First row in the feature map are the first 4 slot of the vector.

By vectorizing the feature maps we transform a 3D conv layer into a 2D matrix of values. Next we need to multiply this matrix by the Transponse of the matrix

The result will be a 8 by 8 matrix. So for example the value in row 4, column 2, will hold the similarities of the fourth and second feature maps in the layer. 
Gram matrix is one of the mostly used in practice when doing style. 

Same as for content, the style loss is calculated. At each of the 5 layers by comparing target vs style image. We only change the target image.

The total loss is basically the sum of the content loss and style loss. We reduce the loss by backpropagation.

However since the style loss and content loss are calculated differently we need to apply constant weights to each. Normally the constant used for style is much larger than the one for content. 

Normally we talk about alfa over beta as ratio.
Example of ratio: 1/10

## PyTorch Implementation of Style Transfer via VGG19
 
```python
# import resources
%matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
```

## Loading VGG19 

```python
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters
for param in vgg.parameters():
    param.requires_grad_(False)
    
# Moving to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
```

##Content and Style Images Loading

```python
def load_image(img_path, max_size=400, shape=None):
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

# load in content and style image
content = load_image('images/IMG-20190704-WA0009.jpg').to(device)
# Resize style to match content
style = load_image('images/gioconda.jpg', shape=content.shape[-2:]).to(device)

```
