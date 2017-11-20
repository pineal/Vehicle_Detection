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

svc = joblib.load('classifier.pkl') 
X_scaler = joblib.load('scaler.pkl') 
print("load classifiers and scalers trained....")

heatmaps = []
labels_cache = []

def pipelines(image): 
    cars_found = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, cars_found)
    heat = apply_threshold(heat, heat_threshold)
    heatmaps.append(heat)
    total_heatmap = sum(heatmaps)
    heatmap = np.clip(total_heatmap, 0, 255)
    labels = label(total_heatmap)
    
    if labels:
        labels_cache.append(labels) 
        final_labels = labels
    else: 
        final_labels = labels_cache[-1]
    
    image = draw_boxes(image, labeled_bboxes(np.copy(image), final_labels), color=(0, 0, 255), thick=6)
    return image

Output_video = 'project_video_output.mp4'
Input_video = 'project_video.mp4'
clip1 = VideoFileClip(Input_video).subclip(38,42)
video_clip = clip1.fl_image(pipelines)
video_clip.write_videofile(Output_video, audio=False)
