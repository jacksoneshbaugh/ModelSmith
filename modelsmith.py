"""
--------------------- ModelSmith --------------------
Create TensorFlow models from YAML config files.
"""

__author__ = 'Jackson Eshbaugh'
__version__ = '12/17/2024'

import numpy as np
import tensorflow as tf
from keras import Input
from keras import layers
from tensorflow.keras import layers, Sequential
import yaml
import pandas as pd


def get_config(file_name):
    """
    Opens the config file and returns the config object.
    :param file_name: the name of the config file
    :return: the config object
    """
    with open(file_name) as f:
        config = yaml.safe_load(f)
        return config


def set_random_seed():
    """
    Sets the numpy random seed based on the config file.
    :return: None
    """
    random_seed = get_config(config_name).get('random_seed', 83)
    np.random.seed(random_seed)


def get_data():
    """
    Finds the data pointed to in the config and splits it into the matrix X (input data) and the vector y (expected
    outputs).
    :return: Matrix X (inputs) and vector y (expected outputs)
    """
    # Access the 'data' section of the YAML file
    config = get_config(config_name).get('data', {})
    if config == {}:
        raise ValueError('Data section missing from config file.')

    if config.get('type') == 'csv':
        # load the CSV
        df = pd.read_csv(config.get('path'))

        inputs = config.get('inputs')
        outputs = config.get('outputs')
        # Select input features (X) based on the 'inputs' list and convert to NumPy array
        X = df[inputs].to_numpy().astype(np.float32)
        # Select output features (y) based on the 'outputs' list and convert to NumPy array
        y = df[outputs].to_numpy().astype(np.float32)
        # Add input columns into X, outputs into y
        return X, y


def create_model():
    """
    Creates the model based on the layers given in the config file.
    :return: the model object
    """
    config = get_config(config_name).get('layers', {})
    model = get_config(config_name).get('model', 'sequential')

    if model.lower() == 'sequential':
        model = Sequential()
    else:
        raise ValueError(f'Model {model} not supported.')
        # Loop through each layer in the YAML file
    for layer_config in config:
        layer_type = layer_config.get('type')
        # Use getattr to dynamically get the layer class by its name
        if layer_type == 'Input':
            shape = layer_config.get('shape')
            layer = Input(shape=shape)  # Input is a special case
        else:
            # Get the layer class dynamically from keras.layers
            layer_class = getattr(layers, layer_type, None)
            if not layer_class:
                raise ValueError(f"Layer type '{layer_type}' not found in Keras.")

            # Initialize the layer with its parameters
            kwargs = {k: v for k, v in layer_config.items() if k != 'type'}
            layer = layer_class(**kwargs)

        # Add the layer to the model
        model.add(layer)
    return model


def train_model(model, X, y):
    """
    Trains the model on the given data.
    :param model: the model object
    :param X: the input data
    :param y: the expected output data
    :return: None
    """
    config = get_config(config_name).get('training', {})
    optimizer_name = config.get('optimizer', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    optimizer = getattr(tf.keras.optimizers, optimizer_name.capitalize(),
                        tf.keras.optimizers.Adam)(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=config.get('loss', 'binary_crossentropy'),
        metrics=config.get('metrics', ['accuracy'])
    )
    model.fit(
        X, y,
        batch_size=config.get('batch_size', 32),
        epochs=config.get('epochs', 10),
        validation_split=config.get('validation_split', 0.2)
    )


if __name__ == "__main__":
    config_name = input('Enter YAML model config file name: ')
    set_random_seed()
    X, y = get_data()
    while True:
        model = create_model()
        model.summary()
        train_model(model, X, y)
        model.summary()
        loss, accuracy = model.evaluate(X, y)
        print(f'Loss: {loss}, Accuracy: {accuracy}')
        response = input('Would you like to save the model? (y/n): ')
        if response == 'y':
            name = input('Please provide a name for the model: ')
            model.save(f'{name}.keras')
            print(f'Model saved as {name}.keras')
        response = input('Would you like to recreate and retrain this model? (y/n): ')
        if response != 'y':
            break
