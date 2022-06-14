# Convolutional Neural Networks for Audio Signal Analysis

This post is an implementation for the [Z by HP Unlocked Challenge 3 - Signal Processing on Kaggle](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing).

## Scope

The scope of this challenge is to be able to identify [Capuchin Bird](https://en.wikipedia.org/wiki/Capuchinbird#/media/File:Capuchinbird_-_Perissocephalus_tricolor.jpg) calls within a given forest audio.

Here is our guy:

![Capuchin Bird](/images/Capuchinbird.jpg)

Picture from Wikipedia

### Plan of attack

The main idea is to obtain a visual representation of the audio into a spectrogram which will then be fed to a CNN that will be able to distinguish whether the image fed in input is a capuchin bird call or something else.

Example of a Spectrogram (Wikipedia)

![Spectrogram](/images/Spectrogram.png)

The challenge is divided into two parts. 
First we will need to be able to build a model that can identify the calls correctly, then we will need to use the model to count the calls within a long forest audio.
The dataset is divided into three splits:

- Parsed Capuchin bird => A collection of capuchin bird calls only. Short clips.
- Parse Non Capuchin bird => A collection of other sounds from the forest/animals. Short clips.
- Forest Recording => A collection of long audios. In these audios there may or may not be capuchin birds calls. Long clips.

## Code Implementation

```python
#Imports

import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
import os, csv
import IPython as IPD
from IPython.display import YouTubeVideo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from itertools import groupby
```
