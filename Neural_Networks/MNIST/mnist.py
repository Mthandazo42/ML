"""
NAME: mnist.py
AUTHOR: MTHANDAZO NDHLOVU
DESCRIPTION: MANIPULATING THE MNIST DATASET, DEEP LEARNING HELLO WORLD SCRIPT
ACKNOWLEDGEMENTS: DEEP LEARNING WITH PYTHON, KERAS, AND STACKOVERFLOW
HAPPY CODING
"""

#IMPORTS
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
import numpy as np

#LOAD THE MNIST DATASET AND THE SPLIT IT TO TRAIN DATA AND TEST DATA
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#RESHAPE THE TRAIN_IMAGES USING THE RESHAPE METHOD AND CHANGE THE DATATYPE TO FLOAT32
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

#DO THE TEST SAME FOR THE TEST IMAGES
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#ARRANGE THE LABELS WELL USING TO_CATEGORICAL UTIL FROM KERAS
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#NOW THAT YOUR DATA IS WELL CLEANED CREATE YOUR NETWORK
#THIS IS SEQUENTIAL NETWORK WITH TWO DENSE LAYERS WITH FIRST ONE HAVING 512 NEURONS AND USING
#RELU AS THE ACTIVATION FUNCTION AND THE LAST ONE HAVING ONLY 10 LAYERS AND USING SOFTMAX
#AS THE ACTIVATION FUNCTION
network = Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#DUE TO A CERTIAN PROBLEM WITH DIRECTLY USING metrics=['accuracy'] I HAD TO REDIFINE THE ACCURACY METRICS
#THANKS TO STACK OVERFLOW GEEKS
#IMPORT THE KERAS BACK
import keras.backend as K

#DEFINE A FUNCTION WHICH SIMPLE ACCEPTS TEST_IMAGES AND TRAIN_LABELS AS INPUT AND RETURNS THE
# KERAS BACKEND MEAN
def get_categorical_accuracy_keras(test_images, train_labels):
    return K.mean(K.equal(K.argmax(test_images, axis=1), K.argmax(train_labels, axis=1)))

#COMPILE YOUR NETWORK
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=[get_categorical_accuracy_keras]) #NOT THE USAGE OF THE FUNCTION DEFINED ABOVE INSTEAD OF THE USUAL 'accuracy'

#FIT THE DATA INTO THE NETWORKS
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#evaluate the test accuracy and test loss of the network model
test_loss, test_acc = network.evaluate(test_images, test_labels)
#print the test accuracy
print('test accuracy: ', test_acc)

