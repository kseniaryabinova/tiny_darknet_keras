from .model import model
from .data_loading import load_data
from .data_preprocessing import preprocess_test_data
from .training import train_model

from . import config

(x_train, y_train), (x_test, y_test) = load_data()
x_test_, y_test_ = preprocess_test_data(x_test, y_test)

if config['model']['is_train']:
    train_model(model, x_train, y_train, x_test_, y_test_)
