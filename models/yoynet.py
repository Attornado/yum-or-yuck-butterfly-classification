from typing import Optional, final
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, EfficientNetV2B2, EfficientNetV2B1, \
    EfficientNetV2B0, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from tensorflow.keras.models import Model
from preprocessing.constants import RANDOM_STATE


__SCALES_EFFICIENT_NET: final = {"b0", "b1", "b2", "b3", "s", "m", "l"}
__POOLS_EFFICIENT_NET: final = {"max", "avg"}
MODEL_NAME_DEFAULT: final = "yoynet"


def YoYNet(input_shape: tuple[Optional[int], Optional[int], Optional[int], Optional[int]], dense_dims: list[int],
           dense_activations: list, dropout_rates: list[float], dense_kernel_regularizers: Optional[list] = None,
           dense_bias_regularizers: Optional[list] = None, dense_activity_regularizers: Optional[list] = None,
           scale: str = "b1", pooling: Optional[str] = None, weights: Optional[str] = 'imagenet',
           name: str = MODEL_NAME_DEFAULT) -> Model:
    """
    Constructs a new YoYNet model, consisting in a convolutional neural network with a pre-trained EfficientNetV2 model
    with a dense classifier on top.

    :param input_shape: 4 elements-long integer tuple representing the input shape of the model.
    :param dense_dims: an integer list containing output dimension for each dense layer at the end of the network.
    :param dense_activations: a list representing the activation function for each dense layer composing the tail of
        the network.
    :param dropout_rates: dropout rates for the hidden dense layers, must be between 0 and 1.
    :param dense_kernel_regularizers: a list representing the kernel regularizers for the dense layers at the end of
        the network (None by default, meaning no kernel regularization is applied).
    :param dense_bias_regularizers: a list representing the bias regularizers for the dense layers at the end of
        the network (None by default, meaning no bias regularization is applied).
    :param dense_activity_regularizers: a list representing the activity regularizers for the dense blocks at the
        end of the network (None by default, meaning no activity regularization is applied).
    :param scale: a string representing the scale of the EfficientNetV2. Either 'b0' for EfficientNetV2B0, 'b1' for
        EfficientNetV2B1, 'b2' for EfficientNetV2B2, 'b3' for EfficientNetV2B3, 's' for EfficientNetV2S, 'm' for
        EfficientNetV2M, 'l' for EfficientNetV2L
    :param pooling: Optional pooling mode for feature extraction. Defaults to None.
      - `None` means that the output of the convolutional blocks will be
          the 4D tensor output of the
          last convolutional layer.
      - `"avg"` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the convolutional blocks will be a 2D tensor.
      - `"max"` means that global max pooling will
          be applied.
    :param weights: either of `None` (random initialization), `"imagenet"` (pre-training on ImageNet), or the path to
        the weights file to be loaded. Defaults to `"imagenet"`.
    :param name: a string representing the name of the model.

    :return: a new YoYNet model with given parameters.

    :raises ValueError: if dense_dims contains any negative number, if len(dense_activations) != len(dense_dims) or
        if dense_kernel_regularizers != None and len(dense_kernel_regularizers) != len(dense_dims) or
        if dense_bias_regularizers != None and len(dense_bias_regularizers) != len(dense_dims) or
        if dense_activity_regularizers != None and len(dense_activity_regularizers) != len(dense_dims) or if the number
        of dropout rates is different from the number of hidden dense layers or if any of the given dropout rates is not
        between 0 and 1 or if the given input shape is incorrect.
    """
    # Do parameter checks
    __check_input_shape(input_shape)
    __check_dense_parameters(
        dense_dims,
        dense_activations,
        dense_kernel_regularizers,
        dense_bias_regularizers,
        dense_activity_regularizers
    )
    __check_dropout_rates(dropout_rates, len(dense_dims))
    __check_scale(scale)
    __check_pooling(pooling)

    # Construct the model
    inputs = tf.keras.Input(shape=input_shape[1:])
    x = inputs
    # x = RandomRotation(factor=0.2, fill_mode='nearest')(x)
    # x = RandomZoom((-0.2, 0.2), fill_mode='nearest')(x)

    # Build EfficientNetV2
    efficient_net = _build_efficient_net(input_shape[1:], scale, pooling, weights)
    x = efficient_net(x)

    # Build dense tail
    x = _build_dense_tail(
        x,
        dense_dims,
        dense_activations,
        dropout_rates,
        dense_kernel_regularizers,
        dense_bias_regularizers,
        dense_activity_regularizers
    )

    # Construct the model
    outputs = x
    model = Model(inputs=inputs, outputs=outputs, name=name)

    return model


def __check_pooling(pooling: Optional[str]):
    """
    Checks if the pooling type is one of the pooling types in the list of pooling types in the EfficientNet model.

    :param pooling: the type of pooling to put at the end of the convolutional blocks, must be either None, 'avg' or
        'max'.
    :type pooling: Optional[str]

    :raise ValueError: if the pooling is not None, 'max' or 'avg'.
    """
    if pooling is not None and pooling not in __POOLS_EFFICIENT_NET:
        raise ValueError(f"pooling type must be one of {list(__POOLS_EFFICIENT_NET)}, got {pooling}")


def __check_scale(scale: str):
    """
    Checks if the scale is either 'b0', 'b1', 'b2', 'b3', 's', 'm' or 'l'.

    :param scale: the scale of the model, should be 'b0', 'b1', 'b2', 'b3', 's', 'm' or 'l'.
    :type scale: str

    :raises ValueError: if scale is not 'b0', 'b1', 'b2', 'b3', 's', 'm' or 'l'.
    """
    if scale not in __SCALES_EFFICIENT_NET:
        raise ValueError(f"Scale must be one of '{list(__SCALES_EFFICIENT_NET)}', got {scale}")


def __check_input_shape(input_shape: tuple[Optional[int], Optional[int], Optional[int], Optional[int]]):
    """
    Checks that the input shape is a 4-tuple of positive integers.

    :param input_shape: the shape of the input tensor.
    :type input_shape: tuple[Optional[int], Optional[int], Optional[int], Optional[int]]

    :raises ValueError: if the input shape is not a 4-tuple of positive integers.
    """
    if len(input_shape) != 4:
        raise ValueError(f"Input shape must be 4-dimensional, got {len(input_shape)}")
    for el in input_shape:
        if el is not None and el <= 0:
            raise ValueError(f"Input shape elements must be positive, got {el}")


def __check_dropout_rates(dropout_rates: list[float], n_dense_layers: int):
    """
    Checks that the dropout rates are between 0 and 1, and that the number of dropout rates is one less than the number
    of dense layers (equal to the number of hidden dense layers).

    :param dropout_rates: dropout rates for the hidden dense layers, must be between 0 and 1.
    :type dropout_rates: list[float]
    :param n_dense_layers: number of dense layers in the model
    :type n_dense_layers: int
    :raise ValueError: if the number of dropout rates is different from the number of hidden dense layers or if any of
        the given dropout rates is not between 0 and 1.
    """
    """
    Checks validity of given dropout_rate.

    :param dropout_rates: dropout rates for the dense layers, must be between 0 and 1 (last excluded).
    :raises ValueError: if dropout_rate < 0 or dropout_rate >= 1.
    """
    if len(dropout_rates) != n_dense_layers - 1:
        raise ValueError(f"dropout_rates must length must match with the number of internal dense layer, got "
                         f"{len(dropout_rates)} and {n_dense_layers - 1}")
    for dropout_rate in dropout_rates:
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"Dropout rates must be between 0 and 1, got {dropout_rate}")


def __check_dense_parameters(dense_dims: list[int], dense_activations: list,
                             dense_kernel_regularizers: Optional[list] = None,
                             dense_bias_regularizers: Optional[list] = None,
                             dense_activity_regularizers: Optional[list] = None):
    """
    Checks validity of dense layer-related parameters.

    :param dense_dims: an integer list containing output dimension for each dense layer at the end of the network.
    :param dense_activations: a list representing the activation function for each dense layer composing the tail of
        the network.
    :param dense_kernel_regularizers: a list representing the kernel regularizers for the dense layers at the end of
        the network (None by default, meaning no kernel regularization is applied).
    :param dense_bias_regularizers: a list representing the bias regularizers for the dense layers at the end of
        the network (None by default, meaning no bias regularization is applied).
    :param dense_activity_regularizers: a list representing the activity regularizers for the dense blocks at the
        end of the network (None by default, meaning no activity regularization is applied).
    :raises ValueError: if dense_dims contains any negative number, if len(dense_activations) != len(dense_dims) or
        if dense_kernel_regularizers != None and len(dense_kernel_regularizers) != len(dense_dims) or
        if dense_bias_regularizers != None and len(dense_bias_regularizers) != len(dense_dims) or
        if dense_activity_regularizers != None and len(dense_activity_regularizers) != len(dense_dims).
    """
    for n in dense_dims:
        if n <= 0:
            raise ValueError(f"Dense layer dimensions must be positive, got {n}")
    if len(dense_activations) != len(dense_dims):
        raise ValueError(f"dense_activations must have the same length as dense_dims, got {len(dense_activations)} "
                         f"and {len(dense_dims)}")
    if dense_kernel_regularizers is not None and len(dense_kernel_regularizers) != len(dense_dims):
        raise ValueError(f"dense_kernel_regularizers must have the same length as dense_dims, got "
                         f"{len(dense_kernel_regularizers)} and {len(dense_dims)}")
    if dense_bias_regularizers is not None and len(dense_bias_regularizers) != len(dense_dims):
        raise ValueError(f"dense_bias_regularizers must have the same length as dense_dims, got "
                         f"{len(dense_bias_regularizers)} and {len(dense_dims)}")
    if dense_activity_regularizers is not None and len(dense_activity_regularizers) != len(dense_dims):
        raise ValueError(f"dense_activity_regularizers must have the same length as dense_dims, got "
                         f"{len(dense_activity_regularizers)} and {len(dense_dims)}")


def _build_efficient_net(input_shape: tuple[Optional[int], Optional[int], Optional[int], Optional[int]], scale: str,
                         pooling: Optional[str], weights: Optional[str]) -> Model:
    """
    Builds an EfficientNetV2 model with the given parameters.

    :param input_shape: 4 elements-long integer tuple representing the input shape of the model.
    :param scale: a string representing the scale of the EfficientNetV2. Either 'b0' for EfficientNetV2B0, 'b1' for
        EfficientNetV2B1, 'b2' for EfficientNetV2B2, 'b3' for EfficientNetV2B3, 's' for EfficientNetV2S, 'm' for
        EfficientNetV2M, 'l' for EfficientNetV2L
    :param pooling: Optional pooling mode for feature extraction. Defaults to None.
      - `None` means that the output of the convolutional blocks will be
          the 4D tensor output of the
          last convolutional layer.
      - `"avg"` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the convolutional blocks will be a 2D tensor.
      - `"max"` means that global max pooling will
          be applied.
    :param weights: either of `None` (random initialization), `"imagenet"` (pre-training on ImageNet), or the path to
        the weights file to be loaded. Defaults to `"imagenet"`.
    :return: an EfficientNetV2 model with the given parameters.
    """
    if scale == 'b0':
        return EfficientNetV2B0(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 'b1':
        return EfficientNetV2B1(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 'b2':
        return EfficientNetV2B2(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 'b3':
        return EfficientNetV2B3(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 's':
        return EfficientNetV2S(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 'm':
        return EfficientNetV2M(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    elif scale == 'l':
        return EfficientNetV2L(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)
    else:
        return EfficientNetV2B1(include_top=False, weights=weights, pooling=pooling, input_shape=input_shape)


def _build_dense_tail(inputs, dense_dims: list[int], dense_activations: list, dropout_rates: list[float],
                      dense_kernel_regularizers: Optional[list] = None, dense_bias_regularizers: Optional[list] = None,
                      dense_activity_regularizers: Optional[list] = None):
    """
   Takes in a list of dense layer dimensions, activations, dropout rates, and regularizers, and returns a Keras model
    with the specified dense layers

    :param dense_dims: an integer list containing output dimension for each dense layer at the end of the network.
    :param dense_activations: a list representing the activation function for each dense layer composing the tail of
        the network.
    :param dropout_rates: dropout rates for the hidden dense layers, must be between 0 and 1.
    :param dense_kernel_regularizers: a list representing the kernel regularizers for the dense layers at the end of
        the network (None by default, meaning no kernel regularization is applied).
    :param dense_bias_regularizers: a list representing the bias regularizers for the dense layers at the end of
        the network (None by default, meaning no bias regularization is applied).
    :param dense_activity_regularizers: a list representing the activity regularizers for the dense blocks at the
        end of the network (None by default, meaning no activity regularization is applied).

    :return: the output of the last layer in the dense tail.
    """
    x = inputs
    for i in range(0, len(dense_dims)):

        if dense_kernel_regularizers is not None:
            dense_kernel_regularizer = dense_kernel_regularizers[i]
        else:
            dense_kernel_regularizer = None
        if dense_bias_regularizers is not None:
            dense_bias_regularizer = dense_bias_regularizers[i]
        else:
            dense_bias_regularizer = None
        if dense_activity_regularizers is not None:
            dense_activity_regularizer = dense_activity_regularizers[i]
        else:
            dense_activity_regularizer = None

        # Instantiate dense layer
        layer = Dense(
            units=dense_dims[i],
            activation=dense_activations[i],
            kernel_regularizer=dense_kernel_regularizer,
            bias_regularizer=dense_bias_regularizer,
            activity_regularizer=dense_activity_regularizer
        )
        x = layer(x)

        # Instantiate dropout if required
        if i < len(dense_dims) - 1 and dropout_rates[i] != 0.0:
            layer = Dropout(rate=dropout_rates[i], seed=RANDOM_STATE)
            x = layer(x)

    return x


