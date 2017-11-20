import glob
import os, sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        zca_whitening=True,
        channel_shift_range=10,
        fill_mode='nearest')

path = './non-vehicles/*'
outPath = './non-vehicles/aug'
os.makedirs(outPath, exist_ok=True)

files = glob.glob(path + '/*.png')
for file in files:
    img = load_img(file)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=outPath, save_prefix='notcar', save_format='png'):
        i += 1
        if i > 1:
            break  # otherwise the generator would loop indefinitely
path = './vehicles/*'
outPath = './vehicles/aug'
os.makedirs(outPath, exist_ok=True)

files = glob.glob(path + '/*.png')
for file in files:
    img = load_img(file)
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=outPath, save_prefix='car', save_format='png'):
        i += 1
        if i > 1:
            break  # otherwise the generator would loop indefinitely