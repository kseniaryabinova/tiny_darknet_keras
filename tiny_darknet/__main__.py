from .model import model
from .data_loading import load_data
from .data_preprocessing import preprocess_train_data, preprocess_test_data
from .callbacks import callbacks_array
from .training import train_model

from . import config

import logging

(x_train, y_train), (x_test, y_test) = load_data()

if config['model']['is_train']:
    datagen, x_train, y_train = preprocess_train_data(x_train, y_train)
    logging.debug(callbacks_array)
    train_model(model, datagen, x_train, y_train, x_test, y_test)

else:
    x_test, y_test = preprocess_test_data(x_test, y_test)
