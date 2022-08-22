import os
import random
from typing import final
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from preprocessing.utils import get_feature_and_label, drop_image_id, train_preprocess, decode_label, decode_image, \
    get_feature
from preprocessing.constants import VALIDATION_SIZE, TRAIN_CSV, TEST_CSV, RANDOM_STATE, EVAL_SIZE, AUTOTUNE, \
    AUGMENTATION_RATIO, BUFFER_SIZE, TRAIN_PATH_CLEANED, EVALUATION_PATH_CLEANED, VALIDATION_PATH_CLEANED, \
    TEST_PATH_CLEANED


# Set random state
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
random.seed = RANDOM_STATE
np.random.seed = RANDOM_STATE
tf.random.set_seed(RANDOM_STATE)


# Preprocessing script constants
_BATCH_SIZE: final = 32


def main():

    # Read csv files
    print("Reading csv files...")
    presplit_csv_data = pd.read_csv(TRAIN_CSV)
    test_csv_data = pd.read_csv(TEST_CSV)

    # Train/eval split (think about eval as labeled test set)
    print("Splitting the dataset into train/test/evaluation/validation "
          "sets and converting them into tensorflow format...")
    train_csv_data, eval_csv_data = train_test_split(
        presplit_csv_data,
        test_size=EVAL_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Train/validation split
    train_csv_data, val_csv_data = train_test_split(
        train_csv_data,
        test_size=VALIDATION_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Get train filenames and labels
    train_images = pd.DataFrame(train_csv_data[['image']].values.tolist())
    train_labels = pd.DataFrame(train_csv_data[['name']].values.tolist())

    # Get evaluation filenames and labels (think about this as the labeled test set)
    eval_images = pd.DataFrame(eval_csv_data[['image']].values.tolist())
    eval_labels = pd.DataFrame(eval_csv_data[['name']].values.tolist())

    # Get unlabeled test filenames
    test_images = pd.DataFrame(test_csv_data[['image']].values.tolist())

    # Get validation filenames and labels
    val_images = pd.DataFrame(val_csv_data[['image']].values.tolist())
    val_labels = pd.DataFrame(val_csv_data[['name']].values.tolist())

    # Convert train set to tensorflow format
    train_ds_prebatch = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds_prebatch = train_ds_prebatch.map(get_feature_and_label, num_parallel_calls=AUTOTUNE)

    # Convert evaluation set into tensorflow format
    eval_ds_prebatch = tf.data.Dataset.from_tensor_slices((eval_images, eval_labels))
    eval_ds_prebatch = eval_ds_prebatch.map(get_feature_and_label, num_parallel_calls=AUTOTUNE)

    # Convert validation set into tensorflow format
    val_ds_prebatch = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds_prebatch = val_ds_prebatch.map(get_feature_and_label, num_parallel_calls=AUTOTUNE)

    # Convert test set to tensorflow format
    test_ds_prebatch = tf.data.Dataset.from_tensor_slices((test_images))
    test_ds_prebatch = test_ds_prebatch.map(get_feature, num_parallel_calls=AUTOTUNE)

    # At this point our datasets have an image, label, and image_id component, so drop image id
    train_ds_prebatch = train_ds_prebatch.map(drop_image_id)
    val_ds_prebatch = val_ds_prebatch.map(drop_image_id)
    # Maybe drop from eval and test sets too?

    # Finish preprocessing of the training set, performing data augmentation
    print("Performing data augmentation...")
    train_ds_prebatch = train_ds_prebatch.repeat(AUGMENTATION_RATIO)
    train_ds_prebatch = train_ds_prebatch.map(train_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds_prebatch = train_ds_prebatch.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=True)

    # Get a batch from the train set to test things up, plotting shapes and one image
    print("Printing shapes to check results...")
    x, y = next(iter(train_ds_prebatch.batch(_BATCH_SIZE)))
    print("Shape input batch: ", x.shape)
    print("Shapes:")
    print(train_ds_prebatch)
    image, label = next(iter(train_ds_prebatch.skip(2)))
    print("image shape:", image.shape)
    print("label shape:", label.shape, "\t\t\t", label)
    print(decode_label(label))
    plt.imshow(decode_image(image))  # for normal run display image
    plt.show()
    # IPython.display.display(decode_image(image))  # for Jupyter/Colab/Kaggle display image

    # Store train/eval/validation pre-batch datasets with tensorflow
    print("Storing results...")
    tf.data.experimental.save(dataset=train_ds_prebatch, path=TRAIN_PATH_CLEANED)
    tf.data.experimental.save(dataset=eval_ds_prebatch, path=EVALUATION_PATH_CLEANED)
    tf.data.experimental.save(dataset=val_ds_prebatch, path=VALIDATION_PATH_CLEANED)
    tf.data.experimental.save(dataset=test_ds_prebatch, path=TEST_PATH_CLEANED)

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
