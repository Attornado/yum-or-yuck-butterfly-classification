from typing import Optional, final
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Functional
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Dense, Dropout, Flatten
from tensorflow.python.layers.normalization import BatchNormalization


AVG_POOL: final = "AVG"
MAX_POOL: final = "MAX"


class PlantNet(Functional):
    """
    This class represents a generic 2D convolutional model for image-based plant disease classification, with an
    AlexNet-like architecture.
    """

    def __init__(self, input_shape: tuple, filters: list[int], kernel_sizes: list[tuple[int, int]],
                 conv_activations: list, conv_strides: list[tuple[int, int]], pool_types: list[Optional[str]],
                 pool_sizes: list[Optional[int]], pool_strides: list[Optional[tuple[int, int]]],
                 batch_normalization: list[bool], dense_dims: list[int], dense_activations: list,
                 dropout_conv: float = 0.0, dropout_dense: float = 0.0,
                 dense_kernel_regularizers: Optional[list] = None, dense_bias_regularizers: Optional[list] = None,
                 dense_activity_regularizers: Optional[list] = None):
        """
        Constructs a new PlantNet instance.

        :param input_shape: input shape of the network.
        :param filters: an integer list containing number of convolutional kernels for each convolutional block.
        :param kernel_sizes: an integer couple list representing kernel size for each convolutional block.
        :param conv_activations: a list indicating activation functions for the convolutional blocks.
        :param conv_strides: an integer couple list representing the strides for each convolutional layer ((1, 1) states
            for no strides).
        :param pool_types: a string/None list representing the pooling layer types, either max or average (None states
            for no pooling layer for the corresponding convolutional block).
        :param pool_sizes: an integer/None list representing the pooling layer size for each convolutional block (None
            states for no pooling layer for the corresponding convolutional block).
        :param pool_strides: an integer couple/None list representing the pooling layer strides for each convolutional
            block (None states for no pooling layer and (1, 1) states for no strides for the corresponding convolutional
            block).
        :param batch_normalization: a boolean list indicating whether or not to add a batch normalization layer after
            each convolutional block.
        :param dense_dims: an integer list containing output dimension for each dense layer at the end of the network.
        :param dense_activations: a list representing the activation function for each dense layer composing the tail of
            the network
        :param dropout_conv: dropout rate for convolutional layers (default 0).
        :param dropout_dense: dropout rate for dense layers, except the last one (default 0).
        :param dense_kernel_regularizers: a list representing the kernel regularizers for the dense layers at the end of
            the network (None by default, meaning no kernel regularization is applied).
        :param dense_bias_regularizers: a list representing the bias regularizers for the dense layers at the end of
            the network (None by default, meaning no bias regularization is applied).
        :param dense_activity_regularizers: a list representing the activity regularizers for the dense blocks at the
            end of the network (None by default, meaning no activity regularization is applied).
        """
        # Call superclass constructor
        super(PlantNet, self).__init__()

        # TODO: add consistency checks to the constructor parameters
        # Do consistency checks on the given parameters

        # Set instance variables
        self.__input_shape = input_shape
        self.__filters = filters
        self.__kernel_sizes = kernel_sizes
        self.__conv_activations = conv_activations
        self.__conv_strides = conv_strides
        self.__pool_types = pool_types
        self.__pool_sizes = pool_sizes
        self.__pool_strides = pool_strides
        self.__batch_normalization = batch_normalization
        self.__dense_dims = dense_dims
        self.__dense_activations = dense_activations
        self.__dropout_conv = dropout_conv
        self.__dropout_dense = dropout_dense
        self.__dense_kernel_regularizers = dense_kernel_regularizers
        self.__dense_bias_regularizers = dense_bias_regularizers
        self.__dense_activity_regularizers = dense_activity_regularizers

        # Build convolutional blocks
        self.__conv_blocks, conv_output_shape = self.__build_conv_blocks()

        # Build dense tail
        self.__dense_blocks = self.__build_dense_blocks(conv_output_shape)

        # Build the model
        self.build(input_shape)

    def __build_conv_blocks(self) -> (list[Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Dropout], tuple):
        """
        Builds the convolutional blocks of the network.

        :return: a list containing the convolutional blocks of the network and the corresponding output shape.
        """
        conv_blocks = []
        conv_output_shape = self.__input_shape

        # Build each convolutional block, computing the output shape step-by-step
        for i in range(0, len(self.__filters)):
            conv_block, conv_output_shape = self.__build_conv_block(index=i, conv_input_shape=conv_output_shape)
            conv_blocks.extend(conv_block)

        return conv_blocks, conv_output_shape

    def __build_conv_block(self, index: int, conv_input_shape: tuple) -> (list[Conv2D, MaxPool2D, AvgPool2D,
                                                                               BatchNormalization, Dropout], tuple):
        """
        Builds the convolutional block corresponding to the given index, composed of a convolutional layer, a pooling
        layer (if specified) and a batch normalization layer (if specified).

        :param index: index of the convolutional block to build.
        :param conv_input_shape: input_shape of the convolutional block.
        :return: a list containing the layers of the convolutional block corresponding to the given index
        """
        conv_block = []
        output_shape = conv_input_shape

        # Add convolutional layer
        convolutional_layer = Conv2D(
            filters=self.__filters[index],
            kernel=self.__kernel_sizes[index],
            strides=self.__conv_strides[index],
            activation=self.__conv_activations[index],
            padding="same",
            name=f"conv_layer{index}"
        )
        output_shape = convolutional_layer.compute_output_shape(output_shape)
        conv_block.append(convolutional_layer)

        # Add batch normalization layer if required
        if self.__batch_normalization[index]:
            batch_normalization_layer = BatchNormalization(name=f"bn_layer{index}")
            output_shape = batch_normalization_layer.compute_output_shape(output_shape)
            conv_block.append(batch_normalization_layer)

        # Add dropout layer if conv dropout is not 0
        if self.__dropout_conv != 0:
            conv_dropout_layer = Dropout(rate=self.__dropout_conv)
            output_shape = conv_dropout_layer.compute_output_shape(output_shape)
            conv_block.append(conv_dropout_layer)

        # Add pooling layer if required
        if self.__pool_sizes[index] is not None:
            pool_layer = None
            if self.__pool_types[index] == AVG_POOL:
                pool_layer = AvgPool2D(self.__pool_sizes[index], strides=self.__pool_strides[index], padding='same')
            elif self.__pool_types[index] == MAX_POOL:
                pool_layer = MaxPool2D(self.__pool_sizes[index], strides=self.__pool_strides[index], padding='same')
            output_shape = pool_layer.compute_output_shape(output_shape)
            conv_block.append(pool_layer)

        return conv_block, output_shape

    def __build_dense_blocks(self, conv_output_shape: tuple) -> list[Flatten, Dense, Dropout]:
        """
        Builds the dense blocks composing the tail of the network.

        :param conv_output_shape: a integer tuple representing the output shape of the convolutional blocks.

        :return: a list containing the dense blocks of the network.
        """
        # TODO: implement this
        pass

    @property
    def filters(self) -> list[int]:
        """
        Retrieves the number of filters for each convolutional block.

        :return: an integer list containing number of convolutional kernels for each convolutional block.
        """
        return self.__filters.copy()

    @property
    def kernel_sizes(self) -> list[tuple[int, int]]:
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
    def conv_strides(self) -> list[tuple[int, int]]:
        """
        Retrieves the strides values for each convolutional layer.

        :return: an integer couple list representing the strides for each convolutional layer ((1, 1) states
            for no strides).
        """
        return self.__conv_strides.copy()

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
    def pool_strides(self) -> list[Optional[tuple[int, int]]]:
        """
        Retrieves the strides values for each pooling layer.

        :return: an integer/None list representing the pooling layer size for each convolutional block (None
            states for no pooling layer for the corresponding convolutional block).
        """
        return self.__pool_strides.copy()

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

    @property
    def dense_kernel_regularizers(self) -> Optional[list]:
        """
        Retrieves the kernel regularizers used in dense blocks.

        :return: a list representing the kernel regularizers for the dense layers at the end of the network, or None if
            no kernel regularization is applied..
        """
        return self.__dense_kernel_regularizers

    @property
    def dense_bias_regularizers(self) -> Optional[list]:
        """
        Retrieves the bias regularizers used in dense blocks.

        :return: a list representing the bias regularizers for the dense layers at the end of the network, or None if
            no bias regularization is applied.
        """
        return self.__dense_bias_regularizers

    @property
    def dense_activity_regularizers(self) -> Optional[list]:
        """
        Retrieves the activity regularizers used in dense blocks.

        :return: a list representing the activity regularizers for the dense layers at the end of the network, or None
            if no activity regularization is applied..
        """
        return self.__dense_activity_regularizers

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
            "conv_strides": self.__conv_strides,
            "pool_types": self.__pool_types,
            "pool_sizes": self.__pool_sizes,
            "pool_strides": self.__pool_strides,
            "batch_normalization": self.__batch_normalization,
            "dense_dims": self.__dense_dims,
            "dense_activations": self.__dense_activations,
            "dropout_conv": self.__dropout_conv,
            "dropout_dense": self.__dropout_dense,
            "dense_kernel_regularizers": self.__dense_kernel_regularizers,
            "dense_bias_regularizers": self.__dense_bias_regularizers,
            "dense_activity_regularizers": self.__dense_activity_regularizers
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
