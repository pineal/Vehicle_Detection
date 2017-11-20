from utils import *
from param import *
# Import all needed modules
import glob
import os
import cv2
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt

print("unit testing....")
# Load the classifier from pickle file
svc = joblib.load('classifier.pkl') 
X_scaler = joblib.load('scaler.pkl') 
print("load classifiers and scalers trained....")

img_index = 0

test_images = glob.glob('./test_images/*.jpg')

for img_src in test_images:
    img = cv2.imread(img_src)
    draw_image = np.copy(img)

    hot_windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    hot_window_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 255), thick=6)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    write_name = './output_images/find_cars' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, hot_window_img)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    write_name = './output_images/heatmap' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, heat)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)
    # Visualize the heatmap when displaying
    heat = np.clip(heat, 0, 255)
    labels = label(heat)
    car_found_img = draw_labeled_bboxes(np.copy(img), labels)
    write_name = './output_images/findcar' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, car_found_img)
    labeled_img = draw_labeled_bboxes(np.copy(img), labels)
    write_name = './output_images/labeled_img' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, labeled_img)
    img_index += 1
