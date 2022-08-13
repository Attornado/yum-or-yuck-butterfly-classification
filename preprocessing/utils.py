import os
from bidict import bidict


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
