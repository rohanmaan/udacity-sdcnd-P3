import os
import numpy as np
import csv
from scipy import misc
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.core import Dense, Activation, Flatten, Dropout
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn
from equalise_function import equalise, transform_image

#name of the home directory
folder = "data/"
#a list of the images in the directory
image_files = os.listdir(folder)
#csv_field_names = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
#initialise empty arrays to store each set of data, image file strings and the float of the steering angle
center_images = []
left_images = []
right_images = []
steering_angles = []
#open the csv file and read through each row appending to the array defined above
with open("data/driving_log.csv") as csvfile:
    print("reading CSV file...")
    reader = csv.DictReader(csvfile)
    for row in reader:
        center_images.append(row['center'])
        left_images.append(row['left'])
        right_images.append(row['right'])
        #to save space convert the string value to a float straight away before appending
        steering_angles.append(float(row['steering']))
    print("reading CSV file complete")
#initialise empty array that will store the loaded and resized files, those same files will be stored in their flipped form

steering_angles, center_images, left_images, right_images = equalise(steering_angles, center_images, left_images, right_images)

center_image_dataset = []
left_image_dataset = []
right_image_dataset = []
center_image_dataset_flip = []
left_image_dataset_flip = []
right_image_dataset_flip = []
center_image_dataset_tranformed = []
left_image_dataset_tranformed = []
right_image_dataset_tranformed = []
center_image_dataset_flip_tranformed = []
left_image_dataset_flip_tranformed = []
right_image_dataset_flip_tranformed = []
#define the resolution the images will be loaded in and what will be the input to the model's first layer
image_size = [80,160,3]
image_size_input = (image_size[0],image_size[1],image_size[2])
print("\nLoading images at size: ", image_size,  "\n")
print("\nLoading center images...\n")
#the reason I'm doing the for<if<else below is due to the naming inconsistency within the csv file for file names
#the csv file contains the original data provided by Udacity and own training data
#personal touch can be seen in my attempts to keep the car centered, but ends up oscillating instead :p
#load and flip the images in taken by the center camera and store in their respective arrays
for img in center_images:
    if img[:4] == "IMG/":
        center_image_dataset.append(misc.imresize(misc.imread("data/" + img), size=image_size))
        center_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img)), size=image_size))
        center_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread("data/" + img)), size=image_size))
        center_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread("data/" + img))), size=image_size))
    else:
        center_image_dataset.append(misc.imresize(misc.imread(img[41:]), size=image_size))
        center_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[41:])), size=image_size))
        center_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread(img[41:])), size=image_size))
        center_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread(img[41:]))), size=image_size))
print("\nLoading center images complete, norm+flip+trans_norm+trans_flip: ", len(center_image_dataset)*4, "\n")
print("\nLoading left images...\n")
#load and flip the images in taken by the left camera and store in their respective arrays
for img in left_images:
    if img[:4] == " IMG":
        left_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
        left_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
        left_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread("data/" + img[1:])), size=image_size))
        left_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread("data/" + img[1:]))), size=image_size))
    else:
        left_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=image_size))
        left_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))
        left_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread(img[42:])), size=image_size))
        left_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread(img[42:]))), size=image_size))
print("\nLoading left images complete, norm+flip+trans_norm+trans_flip: ", len(left_image_dataset)*4, "\n")
print("\nLoading right images...\n")
#load and flip the images in taken by the right camera and store in their respective arrays
for img in right_images:
    if img[:4] == " IMG":
        right_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
        right_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
        right_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread("data/" + img[1:])), size=image_size))
        right_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread("data/" + img[1:]))), size=image_size))
    else:
        right_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=image_size))
        right_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))
        right_image_dataset_tranformed.append(misc.imresize(transform_image(misc.imread(img[42:])), size=image_size))
        right_image_dataset_flip_tranformed.append(misc.imresize(transform_image(np.fliplr(misc.imread(img[42:]))), size=image_size))
print("\nLoading right images complete, norm+flip+trans_norm+trans_flip: ", len(right_image_dataset)*4, "\n")
#transform the arrays into numpy arrays
center_image_dataset = np.asarray(center_image_dataset)
left_image_dataset = np.asarray(left_image_dataset)
right_image_dataset = np.asarray(right_image_dataset)
center_image_dataset_flip = np.asarray(center_image_dataset_flip)
left_image_dataset_flip = np.asarray(left_image_dataset_flip)
right_image_dataset_flip = np.asarray(right_image_dataset_flip)
center_image_dataset_tranformed = np.asarray(center_image_dataset_tranformed)
left_image_dataset_tranformed = np.asarray(left_image_dataset_tranformed)
right_image_dataset_tranformed = np.asarray(right_image_dataset_tranformed)
center_image_dataset_flip_tranformed = np.asarray(center_image_dataset_flip_tranformed)
left_image_dataset_flip_tranformed = np.asarray(left_image_dataset_flip_tranformed)
right_image_dataset_flip_tranformed = np.asarray(right_image_dataset_flip_tranformed)
#initialise empty arrays to process the labels(steering angles) like the pictures above
steering_angles_left = []
steering_angles_right = []
steering_angles_flip = []
steering_angles_left_flip = []
steering_angles_right_flip = []
#the correction for having the left and right cameras at an angle and only one reading provided (+/-)
angle_correction = 0.2
#perform the flip and correction and store each in its respective array
for angle in steering_angles:
    steering_angles_left.append(float(angle + angle_correction))
    steering_angles_right.append(float(angle - angle_correction))
    steering_angles_flip.append(float(-angle))
    steering_angles_left_flip.append(float(-(angle + angle_correction)))
    steering_angles_right_flip.append(float(-(angle - angle_correction)))
#concatenate all the image data as the database input
dataset_inputs = np.concatenate((center_image_dataset, left_image_dataset, right_image_dataset,\
center_image_dataset_flip, left_image_dataset_flip, right_image_dataset_flip,\
center_image_dataset_tranformed, left_image_dataset_tranformed, right_image_dataset_tranformed,\
center_image_dataset_flip_tranformed, left_image_dataset_flip_tranformed,\
right_image_dataset_flip_tranformed), axis=0)
#concatenate all the label(steering_angle) data as the database labels, use the same order as the image concatenation
dataset_labels = np.concatenate((steering_angles, steering_angles_left, steering_angles_right,\
steering_angles_flip, steering_angles_left_flip, steering_angles_right_flip,\
steering_angles, steering_angles_left, steering_angles_right, steering_angles_flip,\
steering_angles_left_flip, steering_angles_right_flip), axis=0)
#shuffle the data prior to concatenation
dataset_inputs, dataset_labels = shuffle(dataset_inputs, dataset_labels)
print("total number of inputs and labels is:", dataset_inputs.shape, dataset_labels.shape)
#crop values for the input images keeping the same suggested ratio in the lesson
crop_t = int((50/160)*image_size[0])
crop_b = int((20/160)*image_size[1])
#define the architecture of the model.
#similar to the Nvidia model in that it is Normalisation followed by convolutions
#followed by flat layers and a single output node
model = Sequential()
#normalisation to a mean of 0 by dividing with pixel value 255 and subtracting 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_size_input))
#crop the input of this layer by crop_t from the top and crop_b from the bottom
model.add(Cropping2D(cropping=((crop_t,crop_b), (0,0))))
#convolution layer with input shape 60,160,3
#Traffic Sign Classifier model adapted to generally flow like the Nvidia model
#normalisation>convolutions>flat>single output
#convolutional layer, filters=32@3x3 stride, border mode is set to valid
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
#ELU activation
model.add(ELU())
#maxpool layer with 2x2 filter
model.add(MaxPooling2D(pool_size = (2,2)))
#convolutional layer, filters=32@3x3 stride
model.add(Convolution2D(32, 3, 3))
#ELU activation
model.add(ELU())
#maxpool layer with 2x2 filter
model.add(MaxPooling2D(pool_size = (2,2)))
#convolutional layer, filters=32@3x3 stride
model.add(Convolution2D(32, 3, 3))
#ELU activation
model.add(ELU())
#flatten the input to 1D
model.add(Flatten())
#drop out layer with keep = 20%
model.add(Dropout(0.20))
#ELU activation
model.add(ELU())
#drop out layer with keep = 50%
model.add(Dropout(0.50))
#ELU activation
model.add(ELU())
#full connected layer to the single output node
model.add(Dense(1))
#mean square error and Adam optimiser used
model.compile(loss='mse', optimizer='adam')
print("fitting the data to the model...")
#fit the model to the given data, validation = 20%, batch = 64, and epochs=3
history = model.fit(dataset_inputs, dataset_labels, batch_size=64, nb_epoch=3, validation_split=0.2)
print("fitting complete.\nsaving the model...")
#save the model.
model.save("model.h5")
print("model saved")
#END
