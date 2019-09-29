from .data_preprocessing import preprocess_train_data
from . import config


def train_model(model, x_train, y_train, x_test, y_test):
    from .callbacks import callbacks_array

    datagen, x_train_, y_train_ = preprocess_train_data(x_train, y_train)

    model.fit_generator(
        datagen.flow(x_train_, y_train_, batch_size=config['model']['batch_size']),
        epochs=config['model']['epochs'],
        validation_data=(x_test, y_test),
        workers=4, callbacks=callbacks_array)
