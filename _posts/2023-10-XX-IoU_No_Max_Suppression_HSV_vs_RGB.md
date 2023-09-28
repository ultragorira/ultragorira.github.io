# What is IoU (Intersection over Union), Non Max Suppression, HSV and RGB...Computer Vision is fun!

There are many concepts in the field of Computer Vision to be aware of. 
If you have worked with images, maybe not as an engineer but say about labelling images, you might have heard of terms such as Intersection over Union, or more commonly abbreviated as IoU, Non Max Suppression, HSV (Hue Saturation Value) or RGB. These are some basic but very important concepts that one should know about when talking about Computer Vision and ML/AI. So let's dive-in and hopefully you will know more about these concepts after reading throu this post. 

# IoU

When a computer vision model does a detection of an object (a class) on an image with a bounding box, that's called a prediction. 
In the field of ML/AI, labels (or ground-truth), are what the correct detection is for that object (class). 

![Prediction_vs_GT](/images/IOU/Prediction.png) 

As shown in the example image above, the blue bounding box is the correct detection (label) and the yellow bounding box is the prediction, what the model is predicting. Note that the yellow box could also be something that a human would annotate as the object class.

In this particular example, the **Intersection** is the area that is intersected, in common, between the two boxes, below marked in purple.

![Intersection](/images/IOU/Intersection.png)

The **Union**, as it can be deduced from the word, it is the area combined of the two bounding boxes. 

![Union](/images/IOU/Union.png)

Mathematically speaking, the IoU formula is just the Intersection/Union. The result will be a value between 0 and 1. 0 means the two boxes do not intersect at all and 1 that they are identical (rare if not impossible). The higher the number the more accurate the prediction by the model, or annotation done by a human, is. When talking about IoU, values over 0.5 are considered as "almost acceptable", over .7 "good" and over 0.9 "great". 


# Non Max Suppression



# HSV vs RGB





