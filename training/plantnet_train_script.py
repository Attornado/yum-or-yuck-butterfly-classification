from typing import final
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.regularizers import l1
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import load_model
from models.butterflynet import ButterflyNet
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, CHANNELS, TRAIN_PATH_IMAGES_LABELS, \
    VALIDATION_PATH_IMAGES_LABELS, NORMALIZATION_CONSTANT
from training.utils import ImageGenerator
from training.constants import FITTED_PLANTNET_DIR, PLOT_DIR


_EPOCHS_LOAD: final = 100
_PLANT_NET_LOAD_PATH: final = FITTED_PLANTNET_DIR + f"/plantnet_{_EPOCHS_LOAD}_epochs_v0.1"


def main():
    # Load the training and validation sets
    train = np.load(TRAIN_PATH_IMAGES_LABELS)
    val = np.load(VALIDATION_PATH_IMAGES_LABELS)
    x_train, y_train = train["images"], train["labels"]
    x_val, y_val = val["images"], val["labels"]

    # Define model parameters
    input_shape = (None, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    filters = [32, 64, 64]
    kernel_sizes = [(3, 3), (3, 3), (3, 3)]
    conv_activations = ["relu", "relu", "relu"]
    conv_strides = [(1, 1), (1, 1), (1, 1)]
    pool_types = ["MAX", "MAX", "MAX"]
    pool_sizes = [(2, 2), (2, 2), (2, 2)]
    pool_strides = [(2, 2), (2, 2), (2, 2)]
    batch_normalization = [False, False, False]
    dense_dims = [1024, 64, 10]
    dense_activations = ['relu', 'relu', 'softmax']
    dropout_conv = 0.25
    dropout_dense = 0.6
    dense_kernel_regularizers = [None, None, None]
    dense_bias_regularizers = [None, None, None]
    dense_activity_regularizers = None

    # Instantiate the model and compile it
    retraining = int(input("Insert 0 for training and 1 for retraining: "))
    if retraining == 0:
        model = ButterflyNet(
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
    else:
        weights_only = int(
            input("Insert 0 to load entire model and 1 to load weights only (architectures must match): ")
        )

        if weights_only == 0:
            model = load_model(_PLANT_NET_LOAD_PATH, custom_objects={
                "ButterflyNet": ButterflyNet
            })
        else:
            model = ButterflyNet(
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
            model.load_weights(_PLANT_NET_LOAD_PATH + "/" + "variables/variables")
            model2 = model = ButterflyNet(
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
            model2.set_weights(model.get_weights())
            del model
            model = model2

    # Print model summary
    model.summary()

    # Set model training parameters
    epochs = 200
    batch_size = 50
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
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, min_delta=0.001, restore_best_weights=True)
    ]
    metrics = [
        SparseCategoricalAccuracy(name='Accuracy')
    ]
    version = 0.4  # For easy saving of multiple model versions

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Instantiate generators
    train_generator = ImageGenerator(
        image_filenames=x_train,
        labels=y_train,
        batch_size=batch_size,
        max_normalization=NORMALIZATION_CONSTANT
    )
    val_generator = ImageGenerator(
        image_filenames=x_val,
        labels=y_val,
        batch_size=batch_size,
        max_normalization=NORMALIZATION_CONSTANT
    )

    # Fit the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Save the model
    if retraining != 0:
        model_name = f"plantnet_{epochs + _EPOCHS_LOAD}_epochs_v{version}"
    else:
        model_name = f"plantnet_{epochs}_epochs_v{version}"
    model.save(f'{FITTED_PLANTNET_DIR}/{model_name}')

    # Save model summary into file to store architecture
    with open(f'{FITTED_PLANTNET_DIR}/{model_name}.txt', 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    # Plot results loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/{model_name}.svg")
    plt.show()


if __name__ == "__main__":
    main()
