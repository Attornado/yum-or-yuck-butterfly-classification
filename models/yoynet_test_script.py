from models.yoynet import YoYNet
from tensorflow.python.keras.regularizers import l1
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import l1
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.constants import IMG_WIDTH, IMG_HEIGHT, CHANNELS


def main():
    # Define model parameters
    input_shape = (None, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    pooling = "max"
    dense_dims = [50, 10]
    dense_activations = ['relu', 'softmax']
    dropout_rates = [0.5]
    dense_kernel_regularizers = [None, l1(1e-5)]
    dense_bias_regularizers = [l1(1e-5), None]
    dense_activity_regularizers = None

    # Instantiate the model and compile it
    model = YoYNet(
        input_shape=input_shape,
        dense_dims=dense_dims,
        dense_activations=dense_activations,
        weights='imagenet',
        scale='b1',
        pooling=pooling,
        dropout_rates=dropout_rates,
        dense_kernel_regularizers=dense_kernel_regularizers,
        dense_bias_regularizers=dense_bias_regularizers,
        dense_activity_regularizers=dense_activity_regularizers
    )
    loss = tf.losses.SparseCategoricalCrossentropy()
    optimizer = Adadelta(
        learning_rate=1.0,
        rho=0.95,
        epsilon=1e-07,
        name="Adadelta"
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=[SparseCategoricalAccuracy()])

    # Print model summary
    model.summary()

    model(np.random.rand(1, *input_shape[1:]))
    model.save("fitted_models/prediction/test")

    model = load_model("fitted_models/prediction/test")
    model.summary()


if __name__ == "__main__":
    main()
