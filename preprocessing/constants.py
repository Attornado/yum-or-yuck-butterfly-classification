import os
from typing import final
import tensorflow as tf


# Dataset path-related constants
DATASET_PATH: final = "data"
DATASET_PATH_CLEANED: final = os.path.join(DATASET_PATH, "cleaned")
DATASET_PATH_ORIGINAL: final = os.path.join(DATASET_PATH, "butterfly_mimics")

TRAIN_PATH_ORIGINAL: final = os.path.join(DATASET_PATH_ORIGINAL, "images")
TRAIN_CSV: final = os.path.join(DATASET_PATH_ORIGINAL, "images.csv")
TRAIN_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "train")

VALIDATION_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "validation")
EVALUATION_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "evaluation")

TEST_PATH_ORIGINAL: final = os.path.join(DATASET_PATH_ORIGINAL, "image_holdouts")
TEST_CSV: final = os.path.join(DATASET_PATH_ORIGINAL, "image_holdouts.csv")
TEST_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "test")

# Other dataset-related constants
CLASS_NAMES: final = ['black', 'monarch', 'pipevine', 'spicebush', 'tiger', 'viceroy']
NAME_TASTES = {
    'black': 'yum',
    'monarch': 'yuck',
    'pipevine': 'yuck',
    'spicebush': 'yum',
    'tiger': 'yum',
    'viceroy': 'yuck'
}
CLASS_COUNT: final = len(CLASS_NAMES)
YUMS: final = ['black', 'spicebush', 'tiger']
YUCKS: final = ['monarch', 'pipevine', 'viceroy']

# Image-related constants
IMG_HEIGHT: final = 224
IMG_WIDTH: final = 224
CHANNELS: final = 3
IMG_SIZE: final = (IMG_HEIGHT, IMG_WIDTH)
NORMALIZATION_CONSTANT: final = 255.0

# Splitting-related constants
VALIDATION_SIZE: final = 0.2
EVAL_SIZE: final = 0.2
RANDOM_STATE: final = 43
AUTOTUNE: final = tf.data.experimental.AUTOTUNE

# Other preprocessing constants
AUGMENTATION_RATIO: final = 3
BUFFER_SIZE: final = 1024
