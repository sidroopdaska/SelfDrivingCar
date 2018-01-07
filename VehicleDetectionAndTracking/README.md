# Vehicle Detection and Tracking

## 1. Project Overview

### 1.1 Goal 

The goal of the project is to create a pipeline to detect and track vehicles in a video. A demo of the pipeline can be found on [Youtube](https://www.youtube.com/watch?v=leUGLrnGym4)

[![Advanced Lane Tracking Video](https://img.youtube.com/vi/leUGLrnGym4/0.jpg)](https://www.youtube.com/watch?v=leUGLrnGym4)

### 1.2 Dependencies

* Python 3.x
* OpenCV 3.x
* NumPy
* Matplotlib (for visualisations and reading images)
* Scikit-Learn
* Scikit-Image
* MoviePy (to process video files)
* Pickle

### 1.3 Project structure

* **rawdata_exploration.ipynb**: Jupyter notebook that performs an initial round of the data set exploratory visualisation and summarisation. Furthermore, this notebook also splits the dataset into the training and test set while taking into account the *time series issues* (i.e. multiple consecutive frames)  
* **build_classifier.ipynb**: Jupyter notebook that performs feature extraction, feature normalisation and trains a Linear SVC to classify vehicles in an image
* **detection_and_tracking.ipynb**: Notebook that creates a pipeline to detect & track vehicles in a video
* **readme_images**: Folder to store images used within this README.md
* **test_images/**: Folder containing a set of images for test purposes
* **rawdata.p**: Pickle file containing the processed raw data set of image paths
* **classifier_data.p**: Pickle file containing the trainied Linear SVC and parameters associated with feature extraction
* **project_video.mp4**: Video over which to test the pipeline
* **project_video_output.mp4**: Resulting output on passing the *project_video* through the pipeline

## 2. Pipeline
* Perform **Histogram of Oriented Gradients (HOG)** feature extraction on a labeled training set of images and train a **Linear SVM classifier**. Additionally, spatially binned color features, as well as histograms of color values are appended to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Resolve the problem of multiple detections and false positives
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.




Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
## 3. Reflection and Future Work
