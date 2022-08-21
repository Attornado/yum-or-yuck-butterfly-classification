from models.plantnet import PlantNet
from tensorflow.python.keras.regularizers import l1
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import l1
import numpy as np
from tensorflow.keras.models import load_model


def main():
    # Define model parameters
    input_shape = (None, 640, 480, 3)
    filters = [16, 32, 64, 128]
    kernel_sizes = [(5, 5), (5, 5), (5, 5), (3, 3)]
    conv_activations = ["relu", "relu", "relu", "relu"]
    conv_strides = [(2, 2), (2, 2), (2, 2), (1, 1)]
    pool_types = ["MAX", "AVG", "MAX", "MAX"]
    pool_sizes = [(5, 5), (5, 5), (5, 5), (3, 3)]
    pool_strides = [(1, 1), (1, 1), (1, 1), (1, 1)]
    batch_normalization = [True, True, True, True]
    dense_dims = [50, 10]
    dense_activations = ['relu', 'softmax']
    dropout_conv = 0.2
    dropout_dense = 0.5
    dense_kernel_regularizers = [None, l1(1e-5)]
    dense_bias_regularizers = [l1(1e-5), None]
    dense_activity_regularizers = None

    # Instantiate the model and compile it
    model = PlantNet(
        input_shape=input_shape,
        filters=filters,
        kernel_sizes=kernel_sizes,
        conv_activations=conv_activations,
        conv_strides=conv_strides,
        pool_types=pool_types,
        pool_sizes=pool_sizes,
        pool_strides=pool_strides,
        batch_normalization=batch_normalization,
        dense_dims=dense_dims,
        dense_activations=dense_activations,
        dropout_conv=dropout_conv,
        dropout_dense=dropout_dense,
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

    model(np.random.rand(1, 640, 480, 3))
    model.save("fitted_models/prediction_models/plantnet/test")

    model = load_model("fitted_models/prediction_models/plantnet/test", custom_objects={"PlantNet": PlantNet})
    model.summary()


if __name__ == "__main__":
    main()
