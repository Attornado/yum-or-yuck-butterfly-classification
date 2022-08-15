from typing import Optional
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Functional
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Dense, Dropout, Layer
from tensorflow.python.layers.normalization import BatchNormalization


class PlantNet(Functional):
    """
    This class represents a generic 2D convolutional model for plant disease classification, with an AlexNet-like
    architecture.
    """
    def __init__(self, input_shape: tuple, filters: list[int], kernel_sizes: list[int], conv_activations: list,
                 pool_types: list[Optional[str]], pool_sizes: list[Optional[int]], batch_normalization: list[bool],
                 dense_dims: list[int], dense_activations: list, dropout_conv: float = 0.0, dropout_dense: float = 0.0):
        """
        Constructs a new PlantNet instance.

        :param input_shape: input shape of the network.
        :param filters: an integer list containing number of convolutional kernels for each convolutional block.
        :param kernel_sizes: an integer list representing kernel size for each convolutional block.
        :param conv_activations: a list indicating activation functions for the convolutional blocks.
        :param pool_types: a string/None list representing the pooling layer types, either max or average (None states
            for no pooling layer for the corresponding convolutional block).
        :param pool_sizes: an integer/None list representing the pooling layer size for each convolutional block (None
            states for no pooling layer for the corresponding convolutional block).
        :param batch_normalization: a boolean list indicating whether or not to add a batch normalization layer after
            each convolutional block.
        :param dense_dims: an integer list containing output dimension for each dense layer at the end of the network.
        :param dense_activations: a list representing the activation function for each dense layer composing the tail of
            the network
        :param dropout_conv: dropout rate for convolutional layers (default 0).
        :param dropout_dense: dropout rate for dense layers (default 0).
        """
        # Call superclass constructor
        super(PlantNet, self).__init__()

        # Set instance variables
        self.__input_shape = input_shape
        self.__filters = filters
        self.__kernel_sizes = kernel_sizes
        self.__conv_activations = conv_activations
        self.__pool_types = pool_types
        self.__pool_sizes = pool_sizes
        self.__batch_normalization = batch_normalization
        self.__dense_dims = dense_dims
        self.__dense_activations = dense_activations
        self.__dropout_conv = dropout_conv
        self.__dropout_dense = dropout_dense

        # Build convolutional blocks
        self.__conv_blocks = self.__build_conv_blocks()

        # Build dense tail
        self.__dense_blocks = self.__build_dense_blocks()

        # Build the model
        self.build(input_shape)

    def __build_conv_blocks(self) -> list[Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Dropout]:
        """
        Builds the convolutional blocks of the network.

        :return: a list containing the convolutional blocks of the network.
        """
        pass

    def __build_dense_blocks(self) -> list[Dense, Dropout]:
        """
        Builds the dense blocks composing the tail of the network.

        :return: a list containing the dense blocks of the network.
        """
        pass

    @property
    def filters(self) -> list[int]:
        """
        Retrieves the number of filters for each convolutional block.

        :return: an integer list containing number of convolutional kernels for each convolutional block.
        """
        return self.__filters.copy()

    @property
    def kernel_sizes(self) -> list[int]:
        """
        Retrieves the kernel size for each convolutional block.

        :return: an integer list containing kernel size for each convolutional block.
        """
        return self.__kernel_sizes.copy()

    @property
    def conv_activations(self) -> list:
        """
        Retrieves the activation function for each convolutional layer.

        :return: a list indicating activation functions for the convolutional blocks.
        """
        return self.__conv_activations.copy()

    @property
    def pool_types(self) -> list[Optional[str]]:
        """
        Retrieves the pool type for each convolutional block (either max or average).

        :return: a string/None list representing the pooling layer types, either max or average (None states
            for no pooling layer for the corresponding convolutional block).
        """
        return self.__pool_types.copy()

    @property
    def pool_sizes(self) -> list[Optional[int]]:
        """
        Retrieves the activation function for each convolutional layer.

        :return: an integer/None list representing the pooling layer size for each convolutional block (None
            states for no pooling layer for the corresponding convolutional block).
        """
        return self.__pool_sizes.copy()

    @property
    def batch_normalization(self) -> list[bool]:
        """
        Retrieves whether or not a batch normalization layer is added to each convolutional block.

        :return: a boolean list indicating whether or not to add a batch normalization layer after
            each convolutional block.
        """
        return self.__batch_normalization.copy()

    @property
    def dense_dims(self) -> list[int]:
        """
        Retrieves the output dimension of each dense layer in the tail of the network.

        :return: an integer list containing output dimension for each dense layer at the end of the network.
        """
        return self.__dense_dims.copy()

    @property
    def dense_activations(self) -> list:
        """
        Retrieves the activation function of each dense layer in the tail of the network.

        :return: an list indicating the activation function for each dense layer at the end of the network.
        """
        return self.__dense_dims.copy()

    @property
    def dropout_conv(self) -> float:
        """
        Retrieves the dropout rate used in convolutional blocks.

        :return: a float indicating the dropout rate used in convolutional blocks.
        """
        return self.__dropout_conv

    @property
    def dropout_dense(self) -> float:
        """
        Retrieves the dropout rate used in dense blocks.

        :return: a float indicating the dropout rate used in dense blocks.
        """
        return self.__dropout_dense

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # Call convolutional blocks
        for layer in self.__conv_blocks:
            x = layer(x)

        # Call dense blocks
        for layer in self.__dense_blocks:
            x = layer(x)

        return x

    def get_config(self):
        config_dict = {
            "input_shape": self.__input_shape,
            "kernel_sizes": self.__kernel_sizes,
            "conv_activations": self.__conv_activations,
            "pool_types": self.__pool_types,
            "pool_sizes": self.__pool_sizes,
            "batch_normalization": self.__batch_normalization,
            "dense_dims": self.__dense_dims,
            "dense_activations": self.__dense_activations,
            "dropout_conv": self.__dropout_conv,
            "dropout_dense": self.__dropout_dense
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)






