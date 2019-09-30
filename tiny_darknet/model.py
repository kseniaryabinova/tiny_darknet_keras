import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from . import config

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same',
                 use_bias=False, input_shape=config['model']['input_shape']))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=16, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', use_bias=False))
model.add(BatchNormalization(momentum=config['model']['momentum']))
model.add(keras.layers.LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=config['model']['num_classes'], kernel_size=(1, 1),
                 strides=1, padding='same', use_bias=False))
model.add(Activation('linear'))

model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(Dense(config['model']['num_classes'], activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(config['model']['learning_rate'],
                               config['model']['decay'])

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()
