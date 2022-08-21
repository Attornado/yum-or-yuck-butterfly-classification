import os
from bidict import bidict
import tensorflow as tf
from preprocessing.constants import NAME_TASTES, CLASS_NAMES


def get_class_names(dataset_path: str) -> bidict[int, str]:
    """
    Gets the image class names reading the dataset path.

    :param dataset_path: path of the training dataset, containing the images splitted in class folders.
    :return: a list containing class names.
    """
    class_names: list[str] = []
    class_names_dict: bidict[int, str] = bidict({})

    if os.path.isdir(dataset_path):

        # For each directory entry, add it to the class name list
        for entry in os.listdir(dataset_path):
            if os.path.isdir(dataset_path + "/" + entry):
                class_names.append(entry)

        # Sort class name list
        class_names.sort()

        # Insert class names into bidict
        for i in range(0, len(class_names)):
            class_names_dict[i] = class_names[i]

        return class_names_dict
    else:
        raise ValueError("dataset_path must be a directory")


def n_images(path: str, classes: bidict[int, str]) -> int:
    """
    Gets the total number of the images in the given path with the given classes.

    :param path: dataset path.
    :param classes: class bidirectional dictionary.
    :return: an integer indicating the total number of images in the given path having the given classes.
    """
    count = 0
    for class_num in classes:
        # Get the full class folder path
        class_path = os.path.join(path, classes[class_num])
        count += len(os.listdir(class_path))

    return count


def decode_image(image):
    # image.shape == tf.TensorShape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    return tf.keras.utils.array_to_img(image.numpy())


def decode_label(label):
    """
    :param label: label to decode.

    :return decoded label.
    """
    # label.shape == tf.TensorShape([class_count])
    return CLASS_NAMES[tf.argmax(label)].numpy().decode('UTF-8')


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
