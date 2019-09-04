from __future__ import print_function

from keras.datasets import mnist
from keras.models import load_model
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = load_model('model/lenet1.h5')

index = np.load('data/index_lenet1.npy')
test_index = np.load('data/index_lenet1_test.npy')
train = x_train[index]
test = x_test[test_index]

confidence_train = []

for i in range(10000):
    sample = np.expand_dims(train[i], 0)
    pred = model.predict(sample)[0]
    label = np.argmax(pred)
    temp = pred[label]
    confidence_train.append(temp)

confidence = np.array(confidence_train)
np.savetxt('data/confidence_train.csv', confidence_train, fmt='%s')


confidence_test = []

for i in range(1000):
    sample = np.expand_dims(test[i], 0)
    pred = model.predict(sample)[0]
    label = np.argmax(pred)
    temp = pred[label]
    confidence_test.append(temp)

confidence = np.array(confidence_test)
np.savetxt('data/confidence_test.csv', confidence_test, fmt='%s')
