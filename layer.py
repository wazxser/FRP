from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model
import keras.backend as K


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = load_model('model/lenet1.h5')

index = np.load('data/index_lenet1.npy')
test_index = np.load('data/index_lenet1_test.npy')
train = x_train[index]
test = x_test[test_index]

input_tensor = model.input
layers = model.layers

for layer in layers:
    if layer.name in ['before_softmax']:
        output = layer.output
        fun = K.function([input_tensor], [output])
        for i in range(1):
            temp = fun([ test[ i*1000 : (i+1)*1000 ] ])[0]
            # temp = temp.T
            if i == 0:
                arr = temp
            else:
                arr = np.append(arr, temp, axis=0)
arr = np.array(arr)
np.savetxt('data/layer_test.csv', arr)


for layer in layers:
    if layer.name in ['before_softmax']:
        output = layer.output
        fun = K.function([input_tensor], [output])
        for i in range(10):
            temp = fun([ train[ i*1000 : (i+1)*1000 ] ])[0]
            # temp = temp.T
            if i == 0:
                arr = temp
            else:
                arr = np.append(arr, temp, axis=0)
arr = np.array(arr)
np.savetxt('data/layer_train.csv', arr)