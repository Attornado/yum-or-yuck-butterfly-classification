import os
from bidict import bidict
import numpy as np
import cv2
from tqdm import tqdm
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, TRAIN_PATH, TEST_PATH, CHANNEL_NUM, TEST_PATH_IMAGES, \
    TRAIN_PATH_IMAGES_LABELS
from preprocessing.utils import get_class_names


def _create_training_data(path: str, classes: bidict[int, str]) -> (np.ndarray, np.ndarray):
    """
    Creates the training data arrays.

    :param path: training dataset path.
    :param classes: class bidirectional dictionary.
    :return: a numpy array with shape (dataset_size, IMG_HEIGHT, IMG_WIDTH, channels) representing the images of the
        train set and a numpy integer array with shape (dataset_size, ) representing the labels.
    """
    images = []
    labels = []

    for class_num in classes:

        # Get the full class folder path
        class_path = os.path.join(path, classes[class_num])

        desc = f"Creating training data (images+labels) for class '{classes[class_num]}' ({class_num})"
        # For each image
        for img in tqdm(os.listdir(class_path), desc=desc):

            # Read, resize and append the image to the training set lists
            img_array = cv2.imread(os.path.join(class_path, img))
            new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
            images.append(new_array)
            labels.append(class_num)

    # Convert lists into numpy arrays
    images = np.reshape(np.array(images), (len(images), IMG_HEIGHT, IMG_WIDTH, CHANNEL_NUM))
    labels = np.array(labels, dtype=int)

    return images, labels


def _create_test_data(path: str) -> np.ndarray:
    """
    Creates the test data arrays.

    :param path: test dataset path.
    :return: a numpy array with shape (dataset_size, IMG_HEIGHT, IMG_WIDTH, channels) representing the images of the
        test set.
    """
    images = []

    # For each image
    for img in tqdm(os.listdir(path), desc="Extracting test data (images)"):

        # Read, resize and append the image to the training set lists
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
        images.append(new_array)

    # Convert lists into numpy arrays
    images = np.reshape(np.array(images), (len(images), IMG_HEIGHT, IMG_WIDTH, CHANNEL_NUM))

    return images


def main():
    # Get class names
    class_names = get_class_names(TRAIN_PATH)

    # Get images and labels from dataset
    images, labels = _create_training_data(TRAIN_PATH, class_names)
    test_images = _create_test_data(TEST_PATH)

    # Store train/test arrays and labels to npz file
    np.savez(file=TRAIN_PATH_IMAGES_LABELS, images=images, labels=labels)
    np.save(file=TEST_PATH_IMAGES, arr=test_images)


if __name__ == "__main__":
    main()
