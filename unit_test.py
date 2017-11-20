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

def image_processing(image):
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop = x_start_stop, y_start_stop = y_start_stop, xy_window=(window_size, window_size), xy_overlap=(overlap, overlap))
    #print(len(windows))
    window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
    #cv2.imshow('window_img',window_img)
    write_name = './output_images/window_img' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, window_img)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    hot_window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #cv2.imshow('window_img',hot_window_img)

    cars_found = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    draw_img = draw_boxes(draw_image, cars_found, color=(0, 0, 255), thick=6)
    #cv2.imshow('find_car',draw_img)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)
    # Visualize the heatmap when displaying
    heat = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    #plt.imshow(heatmap, cmap = 'hot')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    
    #heatmap = apply_threshold(heatmap, 3)
    #plt.imshow(heatmap, cmap = 'hot')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

    #labels = label(heatmap)
    #labeled_img = draw_boxes(img, labeled_bboxes(np.copy(image), labels), color=(255, 0, 255), thick=6)
    #cv2.imshow('labeled',labeled_img)        
    #return labeled_img
# Store the test images directory

test_images = glob.glob('./test_images/*.jpg')


for img_src in test_images:
    img = cv2.imread(img_src)
    draw_image = np.copy(img)
    '''

    windows = slide_window(img, x_start_stop = x_start_stop, y_start_stop = y_start_stop, xy_window=(window_size, window_size), xy_overlap=(overlap, overlap))
    #print(len(windows))
    window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
    #cv2.imshow('window_img',window_img)
    write_name = './output_images/window_img' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, window_img)

    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    hot_window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    write_name = './output_images/hotwindows' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, hot_window_img)
    img = img.astype(np.float32)/255
    '''
    hot_windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    hot_window_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 255), thick=6)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    write_name = './output_images/find_cars' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, hot_window_img)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)
    # Visualize the heatmap when displaying
    heat = np.clip(heat, 0, 255)
    labels = label(heat)
    car_found_img = draw_labeled_bboxes(np.copy(img), labels)
    write_name = './output_images/findcar' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, car_found_img)
    write_name = './output_images/heatmap' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, heat)
    labeled_img = draw_labeled_bboxes(np.copy(img), labels)
    write_name = './output_images/labeled_img' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, labeled_img)
    img_index += 1
