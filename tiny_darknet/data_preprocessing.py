import keras
from keras_preprocessing.image import ImageDataGenerator

from . import config


def preprocess_train_data(x, y):
    y_train = keras.utils.to_categorical(y, config['model']['num_classes'])

    x_train = x.astype('float32')
    x_train /= 255

    datagen = ImageDataGenerator(
        # featurewise_center=False,  # set input mean to 0 over the dataset
        # samplewise_center=False,  # set each sample mean to 0
        # featurewise_std_normalization=False,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=7,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format='channels_last',
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train, augment=True)

    return datagen, x_train, y_train


def preprocess_test_data(x, y):
    y_test = keras.utils.to_categorical(y, config['model']['num_classes'])

    x_test = x.astype('float32')
    x_test /= 255

    return x_test, y_test
