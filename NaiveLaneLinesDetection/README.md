# Naive Lane Lines Detection

The goal of this project is to make a pipeline that finds lane lines on the road through naive Computer Vision techniques.

[//]: # (Image References)

[image1]: ./test_images_output/solidYellowCurve.jpg "solidYellowCurve.jpg"
[image2]: ./test_images_output/solidWhiteCurve.jpg "solidWhiteCurve.jpg"
[image3]: ./test_images_output/solidWhiteRight.jpg "solidWhiteRight.jpg"
[image4]: ./test_images_output/solidYellowCurve2.jpg "solidYellowCurve2.jpg"
[image5]: ./test_images_output/solidYellowLeft.jpg "solidYellowLeft.jpg"
[image6]: ./test_images_output/whiteCarLaneSwitch.jpg "whiteCarLaneSwitch.jpg"

A demo of the pipeline can be found on [Youtube](https://www.youtube.com/watch?v=Nn7XRiKza1c)

[![Naive Lane Tracking Video](https://img.youtube.com/vi/Nn7XRiKza1c/0.jpg)](https://www.youtube.com/watch?v=Nn7XRiKza1c)

## Pipeline Overview

My pipeline comprised of -
1) Changing the color space from RGB to grayscale
2) Performing Gaussian smoothing/blurring to remove noise and spurious gradients
3) Performing edge detection using the Canny Edge Detection Algorithm.
4) Implementing a Polygon ROI mask and using this to restrict the location of the 
 detected edges
5) Applying Hough transform to find the line segments over this masked edge detected image.
6) Reducing jitter and smoothing out the detections through a moving average
7) Drawing out these detected line segments on a blank copy of the original image
8) Performing blending of the original and lines image

Results from each step of the pipeline can be observed in my Jupyter notebook: naive_lane_tracker.ipynb. Furthermore,
please find below the overall results visualised for each of the test images.

![image1] ![image2]
![image3] ![image4]
![image5] ![image6]

## Potential shortcomings of the current pipeline

1. Currently the parameters for the Hough Transform and Canny Edge Detection were chose through
trial and error. Maybe incorporate a better mechanism for hyper parameter selection like k-folds
cross validation, etc.

2. The pipeline needs a mechanism to handle different contrast ratios as is quite evident on moving
from the first two test videos to the challenge video


## Suggestions for possible improvements to the pipeline

1. The smoothing algorithm needs to be refined as it still introduces jitter and has an immensely
long window.

2. Extend this pipeline to handle curvature on the road as opposed to just straight lines. Potential for 
introducing better computer vision and deep learning techniques

