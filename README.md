## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/samples0.jpg
[image2]: ./output_images/samples2.jpg
[image3]: ./output_images/windows0.jpg
[image4]: ./output_images/windows3.jpg
[image5]: ./output_images/findcar3.jpg
[image6]: ./output_images/window2.jpg
[image7]: ./output_images/window3.jpg
[image8]: ./output_images/result3.jpg


[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features can be extracted from the `get_hog_features()` function,
and `extract_features()` function is impin pipelines.py

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

Number of Vehicle Images found is 8792 and number of Non-Vehicle Images found is 8968. They are quite even.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


Due to the limit of cv.imread (didn't use matplot imread, which has the cmap == 'hot' option), I can't display the HOG pictures clearly. However, they were generated like samples1.jpg and samples3.jpg. They could be rendered to have a clear sight.



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally decide to use this combination:
```
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
```
I tried differnt color space and spatial size / spartial bins, they all will affect following results. By trying several times, the combination above was my optimial choice. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


I firstly extract the combinated features for car and not-car both. Then the `X_scaler = StandardScaler().fit(X)` was applied for normalization, and a dataset splition with randomization was done just like previous projects: 
`X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)`

Then I used a Linear SVC to fit the data. 

```
svc = LinearSVC()
svc.fit(X_train, y_train)
```

Here is some result of output:
```
75.97042798995972 Seconds to compute features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
24.64 Seconds to train SVC...
Test Accuracy of SVC =  0.9935
```

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used a multi-scale sliding window search method. Four different window size 64x64, 72x72, 96x96 and 128x128 were tried. Results tend to be good in 96x96 and 128x128. I personally perfer 96x96 size and providing 100 windows in searching one window. 

Other parameters are tuned as listed below, and they were chose considering about result and performance both. 
```
scale = 1.5
overlap = 0.5
ystart = 400
ystop = 656
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code of image processing pipelines could be found from around line 540 to 510 in pipelines.py.

Pictures in the output folder with windows*.jpg are the result of applying sliding window and searched by the classdfier. Here is an example:

![alt text][image4]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After extracting features by using hog sub-sampling and make predictions, the cars were found like findcar*.jpg shown:

![alt text][image5]

A heatmap was also generated: again, they may need to be rendered to see clearly in heapmap*.jpg.

When they were generated, I use `scipy.ndimage.measurements.label()` to detect individual blobs in the heatmap.

I applied a threshold to the heatmaps to filter out the false positives, I have tried several heatmap threshold. I found 3 is best for me to filter out most unnecessary boxes.  Below are the resulting bounding boxes are drawn onto the last frame in the series, these pictures could be found in result*.jpg.

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Even after thresholding heatmaps**, there are still having some false positive. What I can improve most is try to cache the detection box result, and use rolling average to try smoothing. (Done)
Need to tune the parameters better. 