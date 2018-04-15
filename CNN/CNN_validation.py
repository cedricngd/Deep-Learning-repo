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

testx = pd.read_csv("./input/csvTestImages 3360x1024.csv",header=None)
testy = pd.read_csv("./input/csvTestLabel 3360x1.csv",header=None)

# type casts
trainImg = trainImg.values.astype('float32')
trainLab = trainLab.values.astype('int32')-1


# first validation
trainImgVal= trainImg[:2688]
trainLabVal= trainLab[:2688]

trainImg = trainImg[2688:]
trainLab = trainLab[2688:]
"""

# second validation
trainImgVal= trainImg[2688:5376]
trainLabVal= trainLab[2688:5376]

trainImg = np.concatenate((trainImg[:2688],trainImg[5376:]),axis=0)
trainLab = np.concatenate((trainLab[:2688],trainLab[5376:]),axis=0)


# third validation
trainImgVal= trainImg[5376:8065]
trainLabVal= trainLab[5376:8065]

trainImg = np.concatenate((trainImg[:5376],trainImg[8065:]),axis=0)
trainLab = np.concatenate((trainLab[:5376],trainLab[8065:]),axis=0)



# forth validation
trainImgVal= trainImg[8065:10752]
trainLabVal= trainLab[8065:10752]

trainImg = np.concatenate((trainImg[:8065],trainImg[10752:]),axis=0)
trainLab = np.concatenate((trainLab[:8065],trainLab[10752:]),axis=0)



# fifth validation
trainImgVal= trainImg[10752:]
trainLabVal= trainLab[10752:]

trainImg = trainImg[:10752]
trainLab = trainLab[:10752]
"""

# convert to one hot encoded
trainLab=to_one_hot(trainLab,28)
trainLabVal=to_one_hot(trainLabVal,28)

# reshape input images to 32x32
trainImgVal = trainImgVal.reshape([-1, 32, 32,1])
trainImg = trainImg.reshape([-1, 32, 32,1])

trainImg, mean1 = du.featurewise_zero_center(trainImg)
trainImgVal, mean2 = du.featurewise_zero_center(trainImgVal)

# Building convolutional network
startTotal = time.time()
network = input_data(shape=[None, 32, 32, 1], name='input') 			
network = conv_2d(network, 80, 3, activation='relu', regularizer="L2")	
network = max_pool_2d(network, 2) 									
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
network = regression(network, optimizer='sgd', learning_rate=0.01,loss='categorical_crossentropy', name='target')

#model creation
model = tflearn.DNN(network, tensorboard_verbose=0)

#model fitting
model.fit({'input': trainImg}, {'target': trainLab}, n_epoch=30,
        snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')


# Evaluate model
score = model.evaluate(trainImgVal, trainLabVal)
print('Test accuracy: %0.2f%%' % (score[0] * 100))

end = time.time()
print("Computation time: ",end - startTotal)

for i in range(5):
	print('\007')
