from typing import final


# Dataset-related constants
DATASET_PATH: final = "data"
DATASET_PATH_CLEANED: final = DATASET_PATH + "/cleaned"

TRAIN_PATH: final = DATASET_PATH + "/train_images"
TRAIN_PATH_CLEANED: final = DATASET_PATH_CLEANED + "/train"
TRAIN_PATH_IMAGES_LABELS: final = TRAIN_PATH_CLEANED + "/train_images_labels.npz"

TRAIN_CSV_PATH: final = DATASET_PATH + "/train.csv"
TRAIN_CSV_PATH_CLEANED: final = DATASET_PATH_CLEANED + "/train_cleaned.csv"

TEST_PATH: final = DATASET_PATH + "/test_images"
TEST_PATH_CLEANED: final = DATASET_PATH_CLEANED + "/test"
TEST_PATH_IMAGES: final = TEST_PATH_CLEANED + "/test_images.npy"

# Image-related constants
CHANNEL_NUM: final = 3
IMG_HEIGHT: final = 640
IMG_WIDTH: final = 480
