# Advanced Lane Lines Detection

A demo of the pipeline (on both the project_video and challenge_video) can be found on [Youtube](https://www.youtube.com/watch?v=leUGLrnGym4)

[![Advanced Lane Tracking Video](https://img.youtube.com/vi/leUGLrnGym4/0.jpg)](https://www.youtube.com/watch?v=leUGLrnGym4)

## Project Overview

### 1.1 Goal
The goal of this project is to use traditional Computer Vision (i.e. non-machine learning) techniques to develop an advanced and robust algorithm that can detect and track lane boundaries in a video. The pipeline highlighted below was designed to operate under the following scenarios:

* It can detect *exactly* two lane lines, i.e. the left and right lane boundaries of the lane the vehicle is currently driving in.
* It cannot detect adjacent lane lines
* The vehicle must be within a lane and must be aligned along the direction of the lane
* If only one of two lane lines have been successfully detected, then the detection is considered invalid and will be discarded. In this case, the pipeline will instead output a lane line fit (for both left and right) based on the moving average of the previous detections. This is due to the lack of an implementation of the lane approximation function (which is considered as future work).

### 1.2 Dependencies

* Python 3.x
* NumPy
* Matplotlib (for charting and visualising images)
* OpenCV 3.x
* Pickle (for storing the camera calibration matrix and distortion coefficients)
* MoviePy (to process video files)

### 1.3 Project structure

* **lane_tracker.ipynb**: Jupyter notebook with a step-by-step walkthrough of the different components of the pipeline 
* **camera_cal/**: Folder containing a collection of chessboard images used for camera calibration and distortion correction
* **camera_calib.p**: Pickle file containing the saved camera calibration matrix and distortion coefficients
* **test_images/**: Folder containing a set of images for test purposes
* **gif_images/**:
* **readme_images**: Directory to store images used within this README.md
* **challenge_video.mp4**: Video containing uneven road surfaces and non-uniform lighting conditions
* **challenge_video_output.mp4**: Resulting output on passing the challenge_video through the pipeline
* **project_video.mp4**: Video with dark road surfaces and non-uniform lighting conditions
* **project_video_output.mp4**: Resulting output on passing the project_video through the pipeline

### 1.4 Usage

To use the pipeline:
* Run *Section 1* of the notebook titled *Camera Calibration and Distortion correction*. The code snippet here looks to load the *camera_calib.p* pickle file. If this is not found, then the calibration process is initiated using the chessboard images 
located under the *camera_cal/* folder

* Once the calibration process is complete, compile all the cells from *Section 2* through to *Section 9*.
* Finally, download the video to be processed, enter its path in the code snippet in *Section 9* along with the path for the output video and run this cell.

## 2. Pipeline

The various steps invovled in the pipeline are as follows, each of these has also been discussed in more detail in the sub sections below:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### 2.1 Camera calibration and distortion correction

Distortion occurs when the camera maps/transforms the 3D object points to 2D image points. Since this transformation process is imperfect, the apparent shape/size/appearance of some of the objects in the image change i.e. they get distorted. Since we are trying to accurately place the car in the world, look at the curve of a lane and steer in the correct direction, we need to correct for this distortion. Otherwise, our measurements are going to be incorrect.

OpenCV provides three functions, namely, ```cv2.findChessboardCorners```, ```cv2.calibrateCamera``` and ```cv2.undistort```to do just that. Let's unpack this to see how it works:
* Firstly, we define a set of object points that represent inside corners in a set of chessboard images. We then map the object points to images points by using ```cv2.findChessboardCorners```.
* Secondly, we call ```cv2.calibrateCamera``` with this newly created list of object points and image poins to compute the camera calibration matrix and distortion coefficients. These are henceforth stored in a pickle file in the root directory for future use.
* Thirdly, we undistort raw images by passing them into ```cv2.undistort``` along with the two params calculated above.

This process has been visualised below for the reader.

TODO: find chessboard, undistort

### 2.2 Perspective Transformation & ROI selection

Following the distortion correction, an undistorted image undergoes Perspective Transformation which warpes the image
into a *bird's eye view* scene. This makes it easier to detect lane lines and measure their curvature.

* Firstly, we compute the transformation matrix by passing the ```src``` and ```dst``` points into ```cv2.getPerspectiveTransform```. These points are determined empirically with the help of the suite of test images.
* Lastly, the undistorted image is warped by passing it into ```cv2.warpPerspective``` along with the transformation matrix

An example of this has been showcased below for convenience.

TODO: image, source and destinatino points highlighted

### 2.3 Generating a thresholded binary image

This was by far the most involved and challenging step of the pipeline. An overview of the test videos and a quick review of the (U.S. government specifications for highway curvature)[http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC] highlighted that the lane lines are either white / yellow in color. Hence, the aim of this step was to take an undistorted warped image and generate a thresholded binary image that only highlighted the pixels that were likely to be part of the left/right lane lines. Moreover, the thresholding process / mask needed to be robust enough to account for sharp tunrs/ uneven road surfaces and most importantly non-uniform lighting conditions.

Many techniques such a different color space transforms and gradient thresholding were experimented with and as a result the following key insights were derived:
* Different color transforms performed better Now you can see that, the S channel is still doing a fairly robust job of picking up the lines under very different color and contrast conditions, while the other selections look mess
* Need for adaptive thresholding
* Gradient thresholding didnt really give any better perofrmance imporvemet

THe final solytion that was used inthe peoject was an esemble, this gave a 10% improvement in lane line detection
s
### 2.4 Lane Line detection: Sliding Window technique

### 2.5 Lane Line detection: Adaptive Search

### 2.6 Compute the lane line curvature and offset

### 2.7 Metres per piexl

### 2.8 Pipeline

## 3. Reflection and Future Work

