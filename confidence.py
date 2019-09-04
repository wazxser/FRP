from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = load_model('./lenet1.h5')

index = np.load('index_lenet1.npy')
test_index = np.load('index_lenet1_test.npy')
train = x_train[index]
test = x_test[test_index]

confidence = []

for i in range(10000):
    sample = np.expand_dims(train[i], 0)
    pred = model.predict(sample)[0]
    label = np.argmax(pred)
    temp = pred[label]
    confidence.append(temp)

confidence = np.array(confidence)
np.savetxt('./confidence_train.csv', confidence, fmt='%s')
