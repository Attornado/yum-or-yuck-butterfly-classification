import os
import random
import numpy as np
from typing import final
import tensorflow as tf


# Dataset-related constants
DATASET_PATH: final = "data"
DATASET_PATH_CLEANED: final = os.path.join(DATASET_PATH, "cleaned")
DATASET_PATH_ORIGINAL: final = os.path.join(DATASET_PATH, "butterfly_mimics")

TRAIN_PATH_ORIGINAL: final = os.path.join(DATASET_PATH_ORIGINAL, "images")
TRAIN_CSV: final = os.path.join(TRAIN_PATH_ORIGINAL, "images.csv")
TRAIN_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "train")
TRAIN_PATH_IMAGES_LABELS: final = TRAIN_PATH_CLEANED + "/train_images_labels.npz"

VALIDATION_PATH_CLEANED: final = DATASET_PATH_CLEANED + "/validation"
VALIDATION_PATH_IMAGES_LABELS: final = VALIDATION_PATH_CLEANED + "/validation_images_labels.npz"

TRAIN_CSV_PATH: final = DATASET_PATH + "/train.csv"
TRAIN_CSV_PATH_CLEANED: final = DATASET_PATH_CLEANED + "/train_cleaned.csv"

TEST_PATH_ORIGINAL: final = os.path.join(DATASET_PATH_ORIGINAL, "image_holdouts")
TEST_CSV: final = os.path.join(TEST_PATH_ORIGINAL, "images.csv")
TEST_PATH_CLEANED: final = os.path.join(DATASET_PATH_CLEANED, "test")
TEST_PATH_IMAGES: final = TEST_PATH_CLEANED + "/test_images.npy"

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
RANDOM_STATE: final = 42
AUTOTUNE: final = tf.data.experimental.AUTOTUNE

# Set random state
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
random.seed = RANDOM_STATE
np.random.seed = RANDOM_STATE
tf.random.set_seed(RANDOM_STATE)