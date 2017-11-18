from utils import *
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

print("unit testing....")
# Load the classifier from pickle file
svc = joblib.load('classifier.pkl') 
X_scaler = joblib.load('scaler.pkl') 
# Store the test images directory
overlap = 0.5
img_index = 0
ystart = 400
ystop = 656
scale = 1.7

#Define Feature Parameters
color_space = 'YCrCb'
orient = 9 
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

test_images = glob.glob('./test_images/*.jpg')
for img_src in test_images:
    t1 = time.time()
    image = cv2.imread(img_src)
    draw_image = np.copy(image)

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[ystart, ystop], 
                    xy_window=(72, 72), xy_overlap=(overlap, overlap))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    print(time.time() - t1, 'seconds to process one image searching', len(windows), 'windows')

    write_name = './output_images/windows' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, window_img)
    img_index += 1

img_index = 0

def image_processing(img):
    hot_windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_boxes(draw_image, labeled_bboxes(np.copy(image), labels), color=(0, 0, 255), thick=6)
    return draw_img

for img_src in test_images:
    img = cv2.imread(img_src)
    draw_img = image_processing(img)
    write_name = './output_images/result' + str(img_index) + '.jpg'
    cv2.imwrite(write_name, draw_img)
    img_index += 1


Output_video = 'project_video_output.mp4'
#Output_video = 'challenge_video_output.mp4'
#Output_video = 'harder_challenge_video_output.mp4'
Input_video = 'project_video.mp4'
#Input_video = 'challenge_video.mp4'
#Input_video = 'harder_challenge_video.mp4'
clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(image_processing)
video_clip.write_videofile(Output_video, audio=False)
