import os
from typing import final
from bidict import bidict
import tensorflow as tf
from preprocessing.constants import NAME_TASTES, CLASS_NAMES, TRAIN_PATH_ORIGINAL, CHANNELS, IMG_SIZE, \
    NORMALIZATION_CONSTANT, IMG_HEIGHT, IMG_WIDTH, TEST_PATH_ORIGINAL


MAX_DELTA_BRIGHTNESS: final = 32.0
RANDOM_SATURATION_LOWER: final = 0.5
RANDOM_SATURATION_UPPER: final = 1.5


def get_class_names() -> bidict[int, str]:
    """
    Gets the image class names reading the dataset path.

    :return a bidict mapping class names into integer, and viceversa.
    """
    class_names: list[str] = CLASS_NAMES
    class_names_dict: bidict[int, str] = bidict({})
    class_names.sort()

    # Insert class names into bidict
    for i in range(0, len(class_names)):
        class_names_dict[i] = class_names[i]

    return class_names_dict


def decode_image(image):
    """
    It takes a tensor of shape (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) and returns a PIL image.

    :param image: an array of shape (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) representing the image to decode.
    :return: PIL image obtained from given array.
    """
    # image.shape == tf.TensorShape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
    return tf.keras.utils.array_to_img(image.numpy())


def decode_label(label):
    """
    :param label: label to decode.

    :return decoded label.
    """
    # label.shape == tf.TensorShape([class_count])
    classes = get_class_names()
    return classes[label]


def decode_image_id(image_id):
    """
    Decodes the image id.

    :param image_id: encoded image id.

    :return decoded image id.
    """
    # image_id.shape == tf.TensorShape([1])
    return image_id.numpy()[0].decode('UTF-8')


def get_error_type(name: str, predicted_name: str):
    """
    Gets the error type string, either 'TP' (true positive), 'TN' (true negative), 'FP' (false positive) or 'FN' (false
    negative), comparing the real label with the predicted one according to the 'yum' / 'yuck' mapping.

    :param name: real label.
    :param predicted_name: predicted label.

    :return 'TP' if both real and predicted label are a 'yum', 'TN' if both are a 'yuck', 'FP' if real label is a 'yuck'
    and predicted label is a 'yum', and 'FN' if real label is a 'yum' and predicted label is a 'yuck'.
    """
    if NAME_TASTES[name] == 'yum' and NAME_TASTES[predicted_name] == 'yum':
        error_string = 'TP'
    elif NAME_TASTES[name] == 'yum' and NAME_TASTES[predicted_name] == 'yuck':
        error_string = 'FN'
    elif NAME_TASTES[name] == 'yuck' and NAME_TASTES[predicted_name] == 'yum':
        error_string = 'FP'
    elif NAME_TASTES[name] == 'yuck' and NAME_TASTES[predicted_name] == 'yuck':
        error_string = 'TN'
    else:
        error_string = ''

    return error_string


def __get_feature_and_label_function(image_id, class_name):
    """
    It takes in an image id and a class name, reads the image from disk, resizes it, normalizes it, and returns the
    image, the integer label, and the image id.

    :param image_id: the image id of the image we want to load
    :param class_name: the name of the class.
    :return: the image, the integer label, and the image id.
    """
    _image_id = image_id[0].decode('UTF-8')
    _class_name = class_name[0].decode('UTF-8')

    # Get the image
    _img = tf.io.read_file(os.path.join(TRAIN_PATH_ORIGINAL, _image_id + '.jpg'))
    _img = tf.image.decode_jpeg(
        _img,
        channels=CHANNELS,
        dct_method='INTEGER_ACCURATE',
        name=_image_id
    )
    _img = tf.image.resize(_img, IMG_SIZE)
    _img = tf.cast(_img, tf.float32)/NORMALIZATION_CONSTANT

    # Get integer label
    integer_label = get_class_names().inverse[_class_name]

    return _img, integer_label, image_id


def get_feature_and_label(x, y):
    """
    It takes a filename and a label, and returns a tuple of the image, the label, and the filename.

    :param x: the image data.
    :param y: the labels for the images.
    :return: a tuple of 3 tensors:
        1. The image as a float32 tensor;
        2. The label as a int32 tensor;
        3. The image id as a string tensor.
    """

    features_labels = tf.numpy_function(
        __get_feature_and_label_function,
        [x, y],
        [tf.float32, tf.int32, tf.string]
    )

    # numpy_function() loses the shapes, we will need to restore them

    features_labels[0].set_shape(
        tf.TensorShape([IMG_HEIGHT, IMG_WIDTH, CHANNELS])
    )

    features_labels[1].set_shape(tf.TensorShape(()))
    features_labels[2].set_shape(tf.TensorShape([1]))
    tf.cast(features_labels[2], tf.string, name='image_id')

    return features_labels


def __get_feature_function(image_id):
    """
    It reads a test image, resizes it to the desired size, and normalizes it (intended use with tensorflow map
    function).

    :param image_id: the image id of the test image to read.
    :return: the image, a tensor of shape (IMG_HEIGHT, IMG_WIDTH, CHANNELS) and the image id.
    """
    _image_id = image_id[0].decode('UTF-8')
    _img = tf.io.read_file(os.path.join(TEST_PATH_ORIGINAL, _image_id + '.jpg'))
    _img = tf.image.decode_jpeg(
        _img,
        channels=CHANNELS,
        dct_method='INTEGER_ACCURATE',
        name=_image_id
    )
    _img = tf.image.resize(_img, IMG_SIZE)
    _img = tf.cast(_img, tf.float32)/NORMALIZATION_CONSTANT

    return _img, image_id


def get_feature(x):
    """
    It takes a single image as input, and returns a tuple of two elements (intended use with tensorflow map function):

    * The first element is a tensor of shape (IMG_HEIGHT, IMG_WIDTH, CHANNELS) containing the image's pixels.
    * The second element is a tensor of shape (1,) containing the image id.

    :param x: The input tensor.
    :return: a tuple of two elements. The first element is a tensor of shape (IMG_HEIGHT, IMG_WIDTH, CHANNELS) and the
        second element is a tensor of shape (1, ) representing the image id.
    """

    features_labels = tf.numpy_function(
        __get_feature_function,
        [x],
        [tf.float32, tf.string]
    )

    # numpy_function() loses the shapes, we will need to restore them

    features_labels[0].set_shape(
        tf.TensorShape([IMG_HEIGHT, IMG_WIDTH, CHANNELS])
    )

    features_labels[1].set_shape(tf.TensorShape([1]))
    tf.cast(features_labels[1], tf.string, name='image_id')

    return features_labels


def drop_image_id(image_feature, label, image_id):
    """
    Drops an image id from the training set image (intended use with tensorflow map function).

    :param image_feature: the image feature vector
    :param label: The label of the image
    :param image_id: The id of the image

    :return: The image feature and the label.
    """

    return image_feature, label


def train_preprocess(image_feature, label):
    """
    Performs preprocessing for data augmentation purposes (intended use with tensorflow map function).
    Randomly flip the image horizontally/vertically, adjust the brightness and saturation, and make sure the image is
    still in [0, 1] interval.

    :param image_feature: the image feature tensor
    :param label: the label of the image
    :return: the preprocessed image and the label.
    """

    # Randomly flip left/right
    _img = tf.image.random_flip_left_right(image_feature)

    # Randomly flip up/down
    _img = tf.image.random_flip_up_down(_img)

    # Randomly adjust the brightness
    _img = tf.image.random_brightness(_img, max_delta=MAX_DELTA_BRIGHTNESS / NORMALIZATION_CONSTANT)

    # Randomly adjust saturation
    _img = tf.image.random_saturation(_img, lower=RANDOM_SATURATION_LOWER, upper=RANDOM_SATURATION_UPPER)

    # Make sure the image is still in [0, 1]
    _img = tf.clip_by_value(_img, 0.0, 1.0)

    return _img, label
