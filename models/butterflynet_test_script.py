import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.regularizers import l1, l2, L1L2
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from models.butterflynet import ButterflyNet
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, CHANNELS, CLASS_COUNT


def main():
    # Define model parameters
    input_shape = (None, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    pooling = "avg"
    dense_dims = [512, 256, CLASS_COUNT]
    dense_activations = ['relu', 'relu', 'softmax']
    dropout_rates = [0.5, 0.5]
    weights = 'imagenet'
    freeze = False
    version = 'vgg19'
    dense_kernel_regularizers = [l1(1e-5), l1(1e-5), None]
    dense_bias_regularizers = [l1(1e-5), l1(1e-5), None]
    dense_activity_regularizers = [l1(1e-5), l1(1e-5), l1(1e-5)]

    # Instantiate the model and compile it
    model = ButterflyNet(
        input_shape=input_shape,
        dropout_rates=dropout_rates,
        dense_dims=dense_dims,
        dense_activations=dense_activations,
        weights=weights,
        pooling=pooling,
        version=version,
        dense_kernel_regularizers=dense_kernel_regularizers,
        dense_bias_regularizers=dense_bias_regularizers,
        dense_activity_regularizers=dense_activity_regularizers,
        freeze=freeze
    )
    # Print model summary
    model.summary()

    # Set model training parameters
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        name='sparse_categorical_crossentropy'
    )
    optimizer = Adadelta(
        learning_rate=1,
        rho=0.95,
        epsilon=1e-07,
        name='adadelta_optimizer'
    )
    metrics = [
        SparseCategoricalAccuracy(name='accuracy')
    ]

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )


if __name__ == "__main__":
    main()
