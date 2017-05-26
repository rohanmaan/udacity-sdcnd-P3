import os
import numpy as np
import csv
from scipy import misc
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn

folder = "data/"
image_files = os.listdir(folder)
image_size_h = 160
image_size_w = 320

csv_field_names = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
center_images = []
left_images = []
right_images = []
steering_angles = []

samples = []
with open("data/driving_log.csv") as csvfile:

    print("reading CSV file...")

    # reader = csv.DictReader(csvfile)
    # for row in reader:
    #     center_images.append(row['center'])
    #     left_images.append(row['left'])
    #     right_images.append(row['right'])
    #     steering_angles.append(float(row['steering']))

    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

    print("reading CSV file complete")
# print(len(center_images))

print("splitting the samples from: total = ", len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("to train_samples: ", len(train_samples), " and validation_samples: ", len(validation_samples))
print("multi camera & split train_samples: ", len(train_samples)*6, " and validation_samples: ", len(validation_samples)*6)

image_size = [160,320,3]
image_size_input = (image_size[0],image_size[1],image_size[2])
angle_correction = 0.2

# def generator(samples, batch_size=32):
#     # print("accessing generator...\nthe number of input samples is:", len(samples), "and the batch size is:", batch_size)
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates\
#         # print("shuffling samples")
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#
#             images = []
#             angles = []
#             for img in batch_samples:
#                 if (img[0])[:4] == "IMG/":
#                     images.append(misc.imresize(misc.imread("data/" + img[0]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread("data/" + img[0])), size=image_size))
#                     angles.append(float(img[3]))
#                     angles.append(-float(img[3]))
#                 else:
#                     images.append(misc.imresize(misc.imread((img[0])[41:]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread((img[0])[41:])), size=image_size))
#                     angles.append(float(img[3]))
#                     angles.append(-float(img[3]))
#
#                 if (img[1])[:4] == " IMG":
#                     images.append(misc.imresize(misc.imread("data/" + (img[1])[1:]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread("data/" + (img[1])[1:])), size=image_size))
#                     angles.append(float(img[4]) + angle_correction)
#                     angles.append(-float(img[4]) + angle_correction)
#                 else:
#                     images.append(misc.imresize(misc.imread((img[1])[42:]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread((img[1])[42:])), size=image_size))
#                     angles.append(float(img[4]) + angle_correction)
#                     angles.append(-float(img[4]) + angle_correction)
#
#                 if (img[2])[:4] == " IMG":
#                     images.append(misc.imresize(misc.imread("data/" + (img[2])[1:]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread("data/" + (img[2])[1:])), size=image_size))
#                     angles.append(float(img[5]) - angle_correction)
#                     angles.append(-float(img[5]) - angle_correction)
#                 else:
#                     images.append(misc.imresize(misc.imread((img[2])[42:]), size=image_size))
#                     images.append(misc.imresize(np.fliplr(misc.imread((img[2])[42:])), size=image_size))
#                     angles.append(float(img[5]) - angle_correction)
#                     angles.append(-float(img[5]) - angle_correction)
#
#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             # print("generator action complete\nreturning X_train: ", len(X_train), " and y_train:", len(y_train))
#             yield sklearn.utils.shuffle(X_train, y_train)

def generator(samples, batch_size=32):

    print("accessing generator...\nthe number of input samples is:", len(samples), "and the batch size is:", batch_size)

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates\
        # print("shuffling samples")
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset+batch_size]

            for row in batch_samples:
                center_images.append(row[0])
                left_images.append(row[1])
                right_images.append(row[2])
                steering_angles.append(float(row[3]))

            # print("################################################\n\n", len(center_images))
            images = []
            angles = []
            counter = 0
            # for img in batch_samples:
            for img in center_images:
                if img[:4] == "IMG/":
                    images.append(misc.imresize(misc.imread("data/" + img), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread("data/" + img)), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-float(steering_angles[counter]))
                else:
                    images.append(misc.imresize(misc.imread(img[41:]), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread(img[41:])), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-float(steering_angles[counter]))
                counter+=1

            counter = 0
            for img in left_images:
                if img[:4] == " IMG":
                    images.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-(float(steering_angles[counter]) + angle_correction))
                else:
                    images.append(misc.imresize(misc.imread(img[42:]), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-(float(steering_angles[counter]) + angle_correction))
                counter+=1

            counter = 0
            for img in right_images:
                if img[:4] == " IMG":
                    images.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-(float(steering_angles[counter]) - angle_correction))
                else:
                    images.append(misc.imresize(misc.imread(img[42:]), size=image_size))
                    images.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))
                    angles.append(float(steering_angles[counter]))
                    angles.append(-(float(steering_angles[counter]) - angle_correction))
                counter+=1
            # print("################################################\n\n", len(images))
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print("generator action complete\nreturning X_train: ", len(X_train), " and y_train:", len(y_train))
            yield sklearn.utils.shuffle(X_train, y_train)








# center_image_dataset = []
# left_image_dataset = []
# right_image_dataset = []
# center_image_dataset_flip = []
# left_image_dataset_flip = []
# right_image_dataset_flip = []

# image_size = [160,320,3]
# image_size_input = (image_size[0],image_size[1],image_size[2])
# for img in center_images:
#     if img[:4] == "IMG/":
#         center_image_dataset.append(misc.imresize(misc.imread("data/" + img), size=image_size))
#         center_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img)), size=image_size))
#     else:
#         center_image_dataset.append(misc.imresize(misc.imread(img[41:]), size=image_size))
#         center_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[41:])), size=image_size))
# for img in left_images:
#     if img[:4] == " IMG":
#         left_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
#         left_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
#     else:
#         left_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=image_size))
#         left_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))
# for img in right_images:
#     if img[:4] == " IMG":
#         right_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=image_size))
#         right_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread("data/" + img[1:])), size=image_size))
#     else:
#         right_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=image_size))
#         right_image_dataset_flip.append(misc.imresize(np.fliplr(misc.imread(img[42:])), size=image_size))

# center_image_dataset = np.asarray(center_image_dataset)
# left_image_dataset = np.asarray(left_image_dataset)
# right_image_dataset = np.asarray(right_image_dataset)
# center_image_dataset_flip = np.asarray(center_image_dataset_flip)
# left_image_dataset_flip = np.asarray(left_image_dataset_flip)
# right_image_dataset_flip = np.asarray(right_image_dataset_flip)
#
# print("center_image_dataset:", len(center_image_dataset),\
# "\nleft_image_dataset:", len(left_image_dataset),\
# "\nright_image_dataset:", len(right_image_dataset),\
# "\nwith the shape:", center_image_dataset[0].shape)
#
# dataset_inputs = np.concatenate((center_image_dataset, left_image_dataset, right_image_dataset,\
# center_image_dataset_flip, left_image_dataset_flip, right_image_dataset_flip), axis=0)
#
# steering_angles_left = []
# steering_angles_right = []
# steering_angles_flip = []
# steering_angles_left_flip = []
# steering_angles_right_flip = []
# angle_correction = 0.2

# for angle in steering_angles:
#     steering_angles_left.append(float(angle + angle_correction))
#     steering_angles_right.append(float(angle - angle_correction))
#     steering_angles_flip.append(float(-angle))
#     steering_angles_left_flip.append(float(-(angle + angle_correction)))
#     steering_angles_right_flip.append(float(-(angle - angle_correction)))

# dataset_labels = np.concatenate((steering_angles, steering_angles_left, steering_angles_right,\
# steering_angles_flip, steering_angles_left_flip, steering_angles_right_flip), axis=0)

# print("label example:", dataset_labels[100])
# print("inputs shape:",dataset_inputs.shape, \
# "\nlabels shape:", dataset_labels.shape, \
# "\nlabels type:", dataset_labels[0].dtype)
#
# dataset_inputs, dataset_labels = shuffle(dataset_inputs, dataset_labels)
# # dataset_inputs = np.transpose(dataset_inputs,(0,3,2,1))
# # dataset_inputs = dataset_inputs
# # dataset_labels = dataset_labels
# print("database inputs and labels shapes:", dataset_inputs.shape, dataset_labels.shape)
# uniques = np.unique(dataset_labels)
# uniques = len(uniques) + 1
# print("unique cases:", uniques)

##TSC model
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32,32,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(Dropout(0.50))
# model.add(Activation('relu'))
# model.add(Dropout(0.50))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(1))

#Nvidia-ish! model
# model = Sequential()
# model.add(Convolution2D(5, 5, 5, border_mode='valid', input_shape=image_size_input))
# model.add(Activation('relu'))
# model.add(Convolution2D(24, 5, 5))
# model.add(Activation('relu'))
# model.add(Convolution2D(36, 5, 5))
# model.add(Activation('relu'))
# model.add(Convolution2D(48, 5, 5))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 5, 5))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

print("intialising generators...")
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
print("generators initialised")
ch, row, col = 3, 160, 320  # Trimmed image format
crop_h = int((50/160)*image_size[0])
crop_w = int((20/320)*image_size[1])
#other
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_size_input, output_shape=image_size_input))
model.add(Cropping2D(cropping=((crop_h,crop_w), (0,0))))
model.add(Convolution2D(8, 8, 8, subsample=(4, 4), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="valid"))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))

# TODO: Compile and train the model

model.compile(loss='mse', optimizer='adam')

# history = model.fit(dataset_inputs, dataset_labels, batch_size=64, nb_epoch=5, validation_split=0.2)
print("fitting the generators to the model...")
history = model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)*6), validation_data=validation_generator,\
nb_val_samples=(len(validation_samples)*6), nb_epoch=3)

print("fitting complete.\nsaving the model...")
model.save("model.h5")
print("model saved")

#END
