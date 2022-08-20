import tensorflow as tf
from typing import Optional
import numpy as np
import cv2
from preprocessing.constants import IMG_HEIGHT, IMG_WIDTH, CHANNEL_NUM
from tensorflow.keras.utils import Sequence
# from tensorflow.python.keras.utils.data_utils import Sequence


class ImageGenerator(Sequence):
    """
    This class represents an image batch generator, useful to handle huge image datasets that don't fit in main memory.
    Requires the train/validation sets to be organized into two array-like objects containing the filenames and labels,
    respectively.
    """

    def __init__(self, image_filenames, labels, batch_size: int, max_normalization: Optional[int] = None):
        """
        Constructor for the image batch generator.

        :param image_filenames: an array-like objects containing the filenames of the images.
        :param labels: an array-like object representing the labels.
        :param batch_size: the batch size.
        :param max_normalization: max value to use in normalization.
        """
        self.__image_filenames = image_filenames
        self.__labels = labels
        self.__batch_size = batch_size
        self.__max_normalization = max_normalization

    @property
    def image_filenames(self):
        return self.__image_filenames

    @image_filenames.setter
    def image_filenames(self, image_filenames):
        self.__image_filenames = image_filenames

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, labels):
        self.__labels = labels

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self.__batch_size = batch_size

    @property
    def max_normalization(self) -> Optional[int]:
        return self.__max_normalization

    @max_normalization.setter
    def max_normalization(self, max_normalization: Optional[int]):
        self.__max_normalization = max_normalization

    def __len__(self):
        """
        Returns the batch number.

        :return: the batch number, obtained by dividing the the total image number with the batch size.
        """
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

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
            img = cv2.resize(cv2.imread(filename), (IMG_WIDTH, IMG_HEIGHT))
            batch_images.append(img)

        # Normalize batch image if required
        if self.max_normalization is not None:
            batch_images = np.array(batch_images).astype(np.float32) / self.max_normalization
        else:
            batch_images = np.array(batch_images).astype(np.float32)

        return batch_images, np.array(batch_y).astype(np.float32)
