from typing import final
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import load_model
from models.yoynet import YoYNet
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, CHANNELS, TRAIN_PATH_CLEANED, VALIDATION_PATH_CLEANED, \
    EVALUATION_PATH_CLEANED, CLASS_COUNT, AUTOTUNE
from training.constants import FITTED_YOYNET_DIR, PLOT_DIR


_EPOCHS_LOAD: final = 100
_YOYNET_LOAD_PATH: final = os.path.join(FITTED_YOYNET_DIR, f"/plantnet_{_EPOCHS_LOAD}_epochs_v0.1")


def main():
    # Define batch size
    batch_size = 32

    # Load the training and validation sets
    train_ds_prebatch = tf.data.experimental.load(TRAIN_PATH_CLEANED)
    val_ds_prebatch = tf.data.experimental.load(VALIDATION_PATH_CLEANED)
    eval_ds_prebatch = tf.data.experimental.load(EVALUATION_PATH_CLEANED)

    # Fetch the batches for the train, evaluation and validation sets to make training faster
    train_ds_batch = train_ds_prebatch.batch(batch_size)
    train_ds = train_ds_batch.prefetch(AUTOTUNE)

    val_ds_batch = val_ds_prebatch.batch(batch_size)
    val_ds = val_ds_batch.prefetch(AUTOTUNE)

    eval_ds = eval_ds_prebatch.batch(batch_size)

    # Define model parameters
    input_shape = (None, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    pooling = "avg"
    dense_dims = [512, CLASS_COUNT]
    dense_activations = ['relu', 'softmax']
    dropout_rates = [0.5]
    weights = 'imagenet'
    dense_kernel_regularizers = [l1(1e-5), None]
    dense_bias_regularizers = [None, None]
    dense_activity_regularizers = [None, None]

    # Instantiate the model and compile it
    retraining = int(input("Insert 0 for training and 1 for retraining: "))
    if retraining == 0:
        model = YoYNet(
            input_shape=input_shape,
            dropout_rates=dropout_rates,
            dense_dims=dense_dims,
            dense_activations=dense_activations,
            weights=weights,
            pooling=pooling,
            dense_kernel_regularizers=dense_kernel_regularizers,
            dense_bias_regularizers=dense_bias_regularizers,
            dense_activity_regularizers=dense_activity_regularizers
        )
    else:
        weights_only = int(
            input("Insert 0 to load entire model and 1 to load weights only (architectures must match): ")
        )

        if weights_only == 0:
            model = load_model(_YOYNET_LOAD_PATH)
        else:
            model = YoYNet(
                input_shape=input_shape,
                dropout_rates=dropout_rates,
                dense_dims=dense_dims,
                dense_activations=dense_activations,
                weights=weights,
                pooling=pooling,
                dense_kernel_regularizers=dense_kernel_regularizers,
                dense_bias_regularizers=dense_bias_regularizers,
                dense_activity_regularizers=dense_activity_regularizers
            )
            model.load_weights(os.path.join(_YOYNET_LOAD_PATH, "variables", "variables"))

    # Print model summary
    model.summary()

    # Set model training parameters
    epochs = 100
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
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        )
    ]
    metrics = [
        SparseCategoricalAccuracy(name='accuracy')
    ]
    version = 0.2  # For easy saving of multiple model versions

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Fit the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Save the model
    if retraining != 0:
        model_name = f"yoynet_{epochs + _EPOCHS_LOAD}_epochs_v{version}"
    else:
        model_name = f"yoynet_{epochs}_epochs_v{version}"
    model.save(f'{FITTED_YOYNET_DIR}/{model_name}')

    # Save model summary into file to store architecture
    with open(f'{FITTED_YOYNET_DIR}/{model_name}.txt', 'w') as fp:
        model.summary(print_fn=lambda x: fp.write(x + '\n'))

    # Plot results for loss and validation
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/{model_name}_loss.svg")
    plt.show()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/{model_name}_accuracy.svg")
    plt.show()


if __name__ == "__main__":
    main()
