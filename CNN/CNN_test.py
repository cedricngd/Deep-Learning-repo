from __future__ import division, print_function, absolute_import

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tflearn
import tflearn.data_utils as du
from numpy import array
from numpy import argmax
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import time


def to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# read training & testing data
trainImg = pd.read_csv("./input/csvTrainImages 13440x1024.csv",header=None)
trainLab = pd.read_csv("./input/csvTrainLabel 13440x1.csv",header=None)

testImg = pd.read_csv("./input/csvTestImages 3360x1024.csv",header=None)
testLab = pd.read_csv("./input/csvTestLabel 3360x1.csv",header=None)

# train type casts
trainImg = trainImg.values.astype('float32')
trainLab = trainLab.values.astype('int32')-1

# test type casts
testImg = testImg.values.astype('float32')
testLab = testLab.values.astype('int32')-1

# convert to one hot encoded
trainLab=to_one_hot(trainLab,28)
testLab=to_one_hot(testLab,28)

# reshape input images to 32x32
trainImg = trainImg.reshape([-1, 32, 32,1])
testImg = testImg.reshape([-1, 32, 32,1])

trainImg, mean1 = du.featurewise_zero_center(trainImg)
testImg, mean2 = du.featurewise_zero_center(testImg)

# Building convolutional networkD
startTotal = time.time()
network = input_data(shape=[None, 32, 32, 1], name='input') 			# input layer
network = conv_2d(network, 80, 3, activation='relu', regularizer="L2")	# 80 filters 3*3
network = max_pool_2d(network, 2) 										#2*2
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 28, activation='softmax')
network = regression(network, optimizer='momentum', learning_rate=0.01,loss='categorical_crossentropy', name='target')

#model creation
model = tflearn.DNN(network, tensorboard_verbose=0)

#model fitting
model.fit({'input': trainImg}, {'target': trainLab}, n_epoch=30,
        snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

# Evaluate model
score = model.evaluate(testImg, testLab)
print('Test accuracy: %0.2f%%' % (score[0] * 100))

end = time.time()
print("Computation time: ",end - startTotal)

for i in range(5):
	print('\007')
