'''
Collection of helper functions for feature extraction
'''

import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg

def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
    '''
    Draw bounding boxes in an image
    :param img(ndarray): Image
    :param bboxes([2-element tuple]): List of bounding boxes, where each bounding box has the
                form ((x1, y1), (x2, y2))
    :param color(tuple): Color value for the bounding box
    :param thickness(int): Thickness of the box
    :return (ndarray): Updated image
    '''
    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def convert_color(img, cspace='HSV'):
    '''
    Converts the color space of an image
    :param img(ndarray): Image
    :param cspace(string): Color space type
    :return (ndarray): Image with the new color space representation
    '''
    if cspace == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(img)


def bin_spatial(img, size=(32, 32)):
    '''
    Performs spatial binning to reduce the image size and consequently the number
    image pixels
    :param img(ndarray): Image
    :param size((int, int)): New image size
    :return (ndarray): Updated image
    '''
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Computes the histogram of color values
    :param img(ndarray): Image
    :param nbins(int): Number of histogram bins
    :param bins_range(int): Lower and upper range of the bins
    :return (ndarray): Histogram of color values
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     transform_sqrt=False, vis=False, feature_vec=True):
    '''
    Wrapper function to compute the HOG (Histogram of Oriented Gradients) feature descriptor
    :param img(ndarray): Image
    :param orient(int): Number of orientation bins
    :param pix_per_cell(int): Number of pixels per cell
    :param cell_per_block(int): Number of cells per block
    :param transform_sqrt(boolean): Boolean flag to apply power law compression to normalize
                    the image before processing
    :param vis(boolean): Boolean flag to turn HOG visualisation on/off
    :param feature_vec(boolean): Boolean flag on whether to return the unrolled feature vector or not
    :return (ndarray): HOG Feature vector
    '''
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=transform_sqrt,
               visualise=vis,
               feature_vector=feature_vec)


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Function to extract features from a single image/video frame
    :param img(ndarray): Image
    :param cspace(string): Color space in which to extract the features
    :param spatial_size((int, int)): Spatial bin size; param for extracting raw color values
    :param hist_bins(int): Number histogram bins; param for extracting Histogram of color values
    :param orient(int): Number of orientation bins; param for extracting HOG features
    :param pix_per_cell(int): Number of pixels per cell; param for extracting HOG features
    :param cell_per_block(int): Number image channels to use for extracting the HOG features
    :param hog_channel(int): Number of cells per block; param for extracting HOG features
    :param spatial_feat(boolean): Boolean flag to extract the raw color values
    :param hist_feat(boolean): Boolean flag to extract the histogram of color values
    :param hog_feat(boolean): Boolean flag to extract the HOG features
    :return (ndarray): Resulting feature vector
    '''

    # 1) Define an empty list to receive features
    features = []
    # print(img.min(), img.max())
    # 2) Apply color conversion
    feature_img = convert_color(img, cspace=color_space) # returns in 0-255 range
    # print(feature_img.min(), feature_img.max())
    # print(feature_img.dtype)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_img, size=spatial_size)
        # 4) Append features to list
        features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_img, nbins=hist_bins)
        # 6) Append features to list
        features.append(hist_features)

    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_img.shape[2]):
                hog_features.extend(get_hog_features(feature_img[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     transform_sqrt=True,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_img[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block,
                                            transform_sqrt=True,
                                            vis=False, feature_vec=True)

        # 8) Append features to list
        features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(features)

