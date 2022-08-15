import tensorflow as tf
import numpy as np
import cv2
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, CHANNEL_NUM


class ImageGenerator(tf.keras.utils.Sequence):
    """
    This class represents an image batch generator, useful to handle huge image datasets that don't fit in main memory.
    Requires the train/validation sets to be organized into two array-like objects containing the filenames and labels,
    respectively.
    """

    def __init__(self, image_filenames, labels, batch_size):
        """
        Constructor for the image batch generator.

        :param image_filenames: an array-like objects containing the filenames of the images.
        :param labels: an array-like object representing the labels.
        :param batch_size: the batch size.
        """
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the batch number.

        :return: the batch number, obtained by dividing the the total image number with the batch size.
        """
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """
        Retrieves the images and labels corresponding to a batch of the sequence.

        :param idx: the index of the batch to retrieve.
        :return: two numpy arrays, representing respectevely the images and corresponding labels contained in the idx-th
         batch.
        """
        # Get filenames and labels of the batch corresponding to given index
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        batch_x = self.image_filenames[batch_start:batch_end]
        batch_y = self.labels[batch_start:batch_end]

        # Construct the image list
        batch_images = []
        for filename in batch_x:
            img = cv2.resize(cv2.imread(filename), (IMG_WIDTH, IMG_HEIGHT, CHANNEL_NUM))
            batch_images.append(img)
        return np.array(batch_images), np.array(batch_y)