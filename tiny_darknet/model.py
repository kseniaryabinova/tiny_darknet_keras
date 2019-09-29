import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from . import config
import logging


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', input_shape=config['model']['input_shape']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=config['model']['num_classes'], kernel_size=(1, 1), strides=1, padding='same'))
model.add(Activation('linear'))

model.add(Flatten())
model.add(Dense(config['model']['num_classes']))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(config['model']['learning_rate'],
                               config['model']['decay'])

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
