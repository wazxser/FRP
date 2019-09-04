from keras.layers import Dense
from keras.models import Model
import numpy as np
from keras.layers import Input


features_train = np.loadtxt('./layer_train.csv')
features_test = np.loadtxt('./layer_test.csv')

robust_train = np.loadtxt('./robust_lenet1_train.csv')[:10000]
robust_test = np.loadtxt('./robust_lenet1_test.csv')[:1000]

input_shape = (10,)
input_tensor = Input(shape=input_shape)

x = Dense(120, activation='relu', name='fc1')(input_tensor)
x = Dense(84, activation='relu', name='fc2')(x)
x = Dense(10, activation='relu')(x)
x = Dense(1)(x)

model = Model(input_tensor, x)

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath='./features_regress.h5',
                             monitor='val_mean_absolute_error',
                             verbose=1,
                             save_best_only='True',
                             mode='auto',
                             period=1)

model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              metrics=['mae'])

model.fit(features_train, robust_train,
          validation_data=(features_test, robust_test),
          batch_size=256,
          epochs=200,
          callbacks=[checkpoint])
