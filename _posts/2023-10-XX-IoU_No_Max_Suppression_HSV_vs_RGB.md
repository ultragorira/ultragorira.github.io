# What is IoU (Intersection over Union), Non Max Suppression, HSV and RGB...Computer Vision is fun!

There are many concepts in the field of Computer Vision to be aware of. 
If you have worked with images, maybe not as an engineer but say about labelling images, you might have heard of terms such as Intersection over Union, or more commonly abbreviated as IoU, Non Max Suppression, HSV (Hue Saturation Value) or RGB. These are some basic but very important concepts that one should know about when talking about Computer Vision and ML/AI. So let's dive-in and hopefully you will know more about these concepts after reading throu this post. 

# IoU

When a computer vision model does a detection of an object (a class) on an image with a bounding box, that's called a prediction. 
In the field of ML/AI, labels (or ground-truth), are what the correct detection is for that object (class). Labels are used in a supervised training. Based on the accuracy of the prediction against the label, the loss in calculated and from there backpropagated to the neurons in all layers. That's often referret to as gradient descent. 

![Prediction_vs_GT](/images/IOU/Prediction.png) 

As shown in the example image above, the blue bounding box is the correct detection (label) and the yellow bounding box is the prediction, what the model is predicting. Note that the yellow box could also be something that a human would annotate as the object class.

In this particular example, the **Intersection** is the area that is intersected, in common, between the two boxes, below marked in purple.

![Intersection](/images/IOU/Intersection.png)

The **Union**, as it can be deduced from the word, it is the area combined of the two bounding boxes. 

![Union](/images/IOU/Union.png)

Mathematically speaking, the IoU formula is just the Intersection divided by Union. The result will be a value between 0 and 1. Value of 0 means the two boxes do not intersect at all and a value of 1 that they are identical (rare if not impossible). The higher the number the more accurate the prediction by the model, or annotation done by a human, is. When talking about IoU, values over 0.5 are considered as "almost acceptable", over .7 "good" and over 0.9 "great".

# How to calculate Intersection and Union areas

Let's take as example this image (created with Stable Diffusion) with the correct bounding box (Green) and the prediction (Yellow):

![IoU](/images/IOU/SD_Animal.png)

In computer vision the origin of the coordinates starts from left top corner. This means that if were to add coordinates to this image, we would have X = 0 and Y = 0 on the left top corner of the image, and X = 768 and Y = 768 at the bottom right corner (the image is 768 by 768 pixels), like this:

![Coordinates_plan](/images/IOU/SD_Animal_Coordinates.png)

X grows from left to right and Y grows from top to bottom. 

Now keeping this in mind, we want to calculate the area of intersection in this image. Having just the top left corner and the bottom right corner will suffice. If you work with YOLO for example you will need a bit different approach, more on it later. 
In our case then these coordinates would be the green top left corner (Label) and the yellow bottom right corner (prediction), like so:

![Intersection_Area](/images/IOU/Intersection_Area.PNG)

The way we can calculate these coordinates is pretty straight forward. 

For X of the left top corner for the intersection, we will take the MAX X coordinate of the top left corner of each bounding boxes. 
For Y of the left top corner for the intersection, we will do exactly the same, MAX Y coordinate of the top left corner each bounding boxes. 

By looking at the image and the red dot, it is clear that the left top corner of the intersection will be then the X,Y left corner of the green bbox. 

For the X of the bottom right corner for the intersection, we will instead take the MIN X coordinate of the bottom right corner of each bounding boxes.
For the Y of the bottom right corner for the intesection, we will take the MIN Y coordinate f the bottom right corner of each bounding boxes. 

In this case, it will be the yellow bottom right corner. 

Note that that in this particulare case we are getting a pair X,Y from the same origin, same box, but this is not always the case. Example:

![X1Y1X2Y2](/images/IOU/X1Y1X2Y2.PNG)

By looking at the image, to take the coordinates of the yellow area (intersection) the top left corner will be

X1, Y1 = PurpleX, OrangeY

Remember we take the MAX of the X and Y

X2, Y2 = OrangeX, PurpleY. 

Hopefully this concept is clear at this point. 

The Union area is simply just summing up both bounding boxes (taken into condieration to remove the union, else you add it twice). 

Let's do a code implementation of the IoU:

```
import numpy as np

def iou(predictions: np.ndarray, 
        labels: np.ndarray, 
        format: str = "x1y1x2y2"):
    """
    Calculates IoU for x1y1x2y2 or xywh bounding boxes

    Parameters:
        predictions (numpy.ndarray): Predictions done by model
        labels (numpy.ndarray): Ground truth
        format (str): x1y1x2y2 format by default, xywh (YOLO style)

    Returns:
        numpy.ndarray: Intersection over union
    """

    if format == "x1y1x2y2":
        box1_x1 = predictions[:, 0:1]
        box1_y1 = predictions[:, 1:2]
        box1_x2 = predictions[:, 2:3]
        box1_y2 = predictions[:, 3:4]
        box2_x1 = labels[:, 0:1]
        box2_y1 = labels[:, 1:2]
        box2_x2 = labels[:, 2:3]
        box2_y2 = labels[:, 3:4]
    
    elif format == "xywh":
        box1_x1 = predictions[:, 0:1] - predictions[:, 2:3] / 2
        box1_y1 = predictions[:, 1:2] - predictions[:, 3:4] / 2
        box1_x2 = predictions[:, 0:1] + predictions[:, 2:3] / 2
        box1_y2 = predictions[:, 1:2] + predictions[:, 3:4] / 2
        box2_x1 = labels[:, 0:1] - labels[:, 2:3] / 2
        box2_y1 = labels[:, 1:2] - labels[:, 3:4] / 2
        box2_x2 = labels[:, 0:1] + labels[:, 2:3] / 2
        box2_y2 = labels[:, 1:2] + labels[:, 3:4] / 2

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    #clip(0) for when boxes do not intersect
    intersection = np.clip((x2 - x1), 0, None) * np.clip((y2 - y1), 0, None)
    box1_area = np.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = np.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def test_intersection_over_union():
    # Test case 1: Two identical boxes, should result in IOU of 1.0
    box1 = np.array([2, 2, 4, 4])  # Format: [x1, y1, x2, y2]
    box2 = np.array([2, 2, 4, 4])
    iou = iou(np.array([box1]), np.array([box2]), format="x1y1x2y2")
    assert np.allclose(iou, 1.0), f"Test case 1 failed: {iou}"

    # Test case 2: Two non-overlapping boxes, should result in IOU of 0.0
    box3 = np.array([5, 5, 6, 6])
    iou = iou(np.array([box1]), np.array([box3]), format="x1y1x2y2")
    assert np.allclose(iou, 0.0), f"Test case 2 failed: {iou}"

    # Test case 3: Two partially overlapping boxes, should result in a valid IOU value
    box4 = np.array([3, 3, 5, 5])
    iou = iou(np.array([box1]), np.array([box4]), format="x1y1x2y2")
    assert 0.0 <= iou <= 1.0, f"Test case 3 failed: {iou}"

    # Test case 4: Test midpoint format with non-overlapping boxes
    box5_midpoint = np.array([3, 3, 2, 2])  # Format: [x_center, y_center, width, height]
    box6_midpoint = np.array([7, 7, 2, 2])
    iou = iou(np.array([box5_midpoint]), np.array([box6_midpoint]), format="xywh")
    assert np.allclose(iou, 0.0), f"Test case 4 failed: {iou}"

    print("All test cases passed!")

test_intersection_over_union()

```

**OUTPUT: All test cases passed!**

In the code above we first defined the function to calculate IoU as described earlier. In the function however we check the format of the boxes passed in input. 
As explained, we were discussing about x1y1 (top left) and x2y2 bottom right as coordinates, but there are knonwn systems like the one used in YOLO, where the coordinates of a box are given by x center, y center, height and width. For these cases we would need to convert the coordinates to that format, as it happens in the function. 

When calculating the intersection, we also need to consider edge cases when there is no intersection at all, hence the clipping to 0. 

Lastly, the IoU is calculated by dividing the intersection with the union (sum of the boxes areas minus intersection to not have it twice).
The + 1e-6 term added to the denominator in the intersection over union (IoU) calculation is used to prevent division by zero. It is a small epsilon value added to the denominator to ensure numerical stability. This is commonly referred to as "smoothing" or "epsilon smoothing."

A function for testing the IoU is then invoked to verify that the IoU calculation is working as expected. 
Testing your code is probably one of the best thing you can do. Creating test cases is time consuming, but nowadays with ChatGPT and other LLMs you can do this in a blink of an eye.

# Non Max Suppression



# HSV vs RGB





