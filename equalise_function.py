import os
import numpy as np
import csv
import cv2
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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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

def equalise(steering_angles, center_images, left_images, right_images):
    print(max(steering_angles), min(steering_angles))
    mu, sigma = 0, 15
    # the histogram of the data
    n, bins, patches = plt.hist(steering_angles, 20, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.xlabel('Turning Angle')
    plt.ylabel('Occurence %')
    plt.axis([-1.1, 1.1, 0, 10])
    plt.grid(True)
    # plt.show()
    # plt.save("initial_occurence.jpg")

    occurence_array = [0 for i in range(20)]
    steering_angles, center_images, left_images, right_images = shuffle(steering_angles, center_images, left_images, right_images)
    range_indexes_1 = []
    range_indexes_2 = []
    range_indexes_3 = []
    range_indexes_4 = []
    range_indexes_5 = []
    range_indexes_6 = []
    range_indexes_7 = []
    range_indexes_8 = []
    range_indexes_9 = []
    range_indexes_10 = []
    range_indexes_11 = []
    range_indexes_12 = []
    range_indexes_13 = []
    range_indexes_14 = []
    range_indexes_15 = []
    range_indexes_16 = []
    range_indexes_17 = []
    range_indexes_18 = []
    range_indexes_19 = []
    range_indexes_20 = []
    for i in range(len(steering_angles)):
        if steering_angles[i] >= -1 and steering_angles[i] < -0.9:
            occurence_array[0] += 1
            range_indexes_1.append(i)
        elif steering_angles[i] >= -0.9 and steering_angles[i] < -0.8:
            occurence_array[1] += 1
            range_indexes_2.append(i)
        elif steering_angles[i] >= -0.8 and steering_angles[i] < -0.7:
            occurence_array[2] += 1
            range_indexes_3.append(i)
        elif steering_angles[i] >= -0.7 and steering_angles[i] < -0.6:
            occurence_array[3] += 1
            range_indexes_4.append(i)
        elif steering_angles[i] >= -0.6 and steering_angles[i] < -0.5:
            occurence_array[4] += 1
            range_indexes_5.append(i)
        elif steering_angles[i] >= -0.5 and steering_angles[i] < -0.4:
            occurence_array[5] += 1
            range_indexes_6.append(i)
        elif steering_angles[i] >= -0.4 and steering_angles[i] < -0.3:
            occurence_array[6] += 1
            range_indexes_7.append(i)
        elif steering_angles[i] >= -0.3 and steering_angles[i] < -0.2:
            occurence_array[7] += 1
            range_indexes_8.append(i)
        elif steering_angles[i] >= -0.2 and steering_angles[i] < -0.1:
            occurence_array[8] += 1
            range_indexes_9.append(i)
        elif steering_angles[i] >= -0.1 and steering_angles[i] < 0.0:
            occurence_array[9] += 1
            range_indexes_10.append(i)
        elif steering_angles[i] >= 0.0 and steering_angles[i] < 0.1:
            occurence_array[10] += 1
            range_indexes_11.append(i)
        elif steering_angles[i] >= 0.1 and steering_angles[i] < 0.2:
            occurence_array[11] += 1
            range_indexes_12.append(i)
        elif steering_angles[i] >= 0.2 and steering_angles[i] < 0.3:
            occurence_array[12] += 1
            range_indexes_13.append(i)
        elif steering_angles[i] >= 0.3 and steering_angles[i] < 0.4:
            occurence_array[13] += 1
            range_indexes_14.append(i)
        elif steering_angles[i] >= 0.4 and steering_angles[i] < 0.5:
            occurence_array[14] += 1
            range_indexes_15.append(i)
        elif steering_angles[i] >= 0.5 and steering_angles[i] < 0.6:
            occurence_array[15] += 1
            range_indexes_16.append(i)
        elif steering_angles[i] >= 0.6 and steering_angles[i] < 0.7:
            occurence_array[16] += 1
            range_indexes_17.append(i)
        elif steering_angles[i] >= 0.7 and steering_angles[i] < 0.8:
            occurence_array[17] += 1
            range_indexes_18.append(i)
        elif steering_angles[i] >= 0.8 and steering_angles[i] < 0.9:
            occurence_array[18] += 1
            range_indexes_19.append(i)
        elif steering_angles[i] >= 0.9 and steering_angles[i] <= 1.0:
            occurence_array[19] += 1
            range_indexes_20.append(i)

    print(min(occurence_array))
    limit = min(occurence_array)

    print(len(range_indexes_1))
    range_indexes_1 = range_indexes_1[:(limit)]
    print(len(range_indexes_1))
    print(len(range_indexes_2))
    range_indexes_2 = range_indexes_2[:(limit)]
    print(len(range_indexes_2))
    range_indexes_3 = range_indexes_3[:(limit)]
    range_indexes_4 = range_indexes_4[:(limit)]
    range_indexes_5 = range_indexes_5[:(limit)]
    range_indexes_6 = range_indexes_6[:(limit)]
    range_indexes_7 = range_indexes_7[:(limit)]
    range_indexes_8 = range_indexes_8[:(limit)]
    range_indexes_9 = range_indexes_9[:(limit)]
    range_indexes_10 = range_indexes_10[:(limit)]
    range_indexes_11 = range_indexes_11[:(limit)]
    range_indexes_12 = range_indexes_12[:(limit)]
    range_indexes_13 = range_indexes_13[:(limit)]
    range_indexes_14 = range_indexes_14[:(limit)]
    range_indexes_15 = range_indexes_15[:(limit)]
    range_indexes_16 = range_indexes_16[:(limit)]
    range_indexes_17 = range_indexes_17[:(limit)]
    range_indexes_18 = range_indexes_18[:(limit)]
    range_indexes_19 = range_indexes_19[:(limit)]
    range_indexes_20 = range_indexes_20[:(limit)]

    indexes = np.concatenate((range_indexes_1, range_indexes_2, range_indexes_3, range_indexes_4,\
    range_indexes_5, range_indexes_6, range_indexes_7, range_indexes_8, range_indexes_9, range_indexes_10,\
    range_indexes_11, range_indexes_12, range_indexes_13, range_indexes_14, range_indexes_15, range_indexes_16,\
    range_indexes_17, range_indexes_18, range_indexes_19, range_indexes_20),axis=0)

    new_center_images = []
    new_left_images = []
    new_right_images = []
    new_steering_angles = []
    for index in indexes:
        new_center_images.append(center_images[index])
        new_left_images.append(left_images[index])
        new_right_images.append(right_images[index])
        new_steering_angles.append(steering_angles[index])

    center_images = new_center_images
    left_images = new_left_images
    right_images = new_right_images
    steering_angles = new_steering_angles

    mu, sigma = 0, 15
    # the histogram of the data
    n, bins, patches = plt.hist(steering_angles, 20, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('Turning Angle')
    plt.ylabel('Occurence %')
    plt.axis([-1.1, 1.1, 0, 5])
    plt.grid(True)
    # plt.show()
    # plt.save("after_occurence.jpg")

    return steering_angles, center_images, left_images, right_images


equalise(steering_angles, center_images, left_images, right_images)


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(image,ang_range = 60,shear_range = 6,trans_range = 20):

    #caused alot of problems?
    
    # Rotation
    # ang_rot = np.random.uniform(ang_range)-ang_range/2
    # rows,cols,ch = image.shape
    # Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    # # Translation
    # tr_x = trans_range*np.random.uniform()-trans_range/2
    # tr_y = trans_range*np.random.uniform()-trans_range/2
    # Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # # Shear
    # pts1 = np.float32([[5,5],[20,5],[5,20]])
    # pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    # pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    # pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    # shear_M = cv2.getAffineTransform(pts1,pts2)
    #
    # image = cv2.warpAffine(image,Rot_M,(cols,rows))
    # image = cv2.warpAffine(image,Trans_M,(cols,rows))
    # image = cv2.warpAffine(image,shear_M,(cols,rows))

    #Brightness augmentation
    image = augment_brightness_camera_images(image)

    return image
