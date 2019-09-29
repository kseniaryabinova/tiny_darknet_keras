from . import config


def train_model(model, datagen, x_train, y_train, x_test, y_test):
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=config['model']['batch_size']),
        epochs=config['model']['epochs'],
        validation_data=(x_test, y_test),
        workers=4)
