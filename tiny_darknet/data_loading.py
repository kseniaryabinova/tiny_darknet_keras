import logging
from keras.datasets import cifar100


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    logging.warning('CIFAR100 has been loaded! \n'
                    '\tTrain has %d examples\n'
                    '\tTest has %d examples',
                    y_train.shape[0], y_test.shape[0])

    return (x_train, y_train), (x_test, y_test)
