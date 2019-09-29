import logging
import keras

from . import config


callbacks_array = []


def callback_decorator(func):
    def function_wrapper():
        logging.info("Register callback: %s", func.__name__)
        callback = func()
        callbacks_array.append(callback)

    return function_wrapper


@callback_decorator
def model_checkpoint():
    callback = keras.callbacks.callbacks.ModelCheckpoint(
        config['model']['path_to_checkpoints'], monitor='val_acc', verbose=1,
        save_best_only=False, save_weights_only=False, period=1)

    return callback


@callback_decorator
def tensorboard():
    callback = keras.callbacks.tensorboard_v1.TensorBoard(
        log_dir=config['model']['path_to_tensorboard_logs'],
        histogram_freq=0, batch_size=config['model']['batch_size'],
        write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
        embeddings_layer_names=None, embeddings_metadata=None,
        embeddings_data=None, update_freq='epoch')

    return callback


tensorboard()
model_checkpoint()
