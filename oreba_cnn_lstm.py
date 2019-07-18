"""CNN-LSTM Model"""

import tensorflow as tf

SCOPE = "oreba_cnn_lstm"


def channels_axis(inputs, data_format):
    """Return the axis index which houses the channels."""
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = len(inputs.get_shape()) - 1
    return axis


def batch_norm(inputs, data_format, name):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=channels_axis(inputs, data_format), center=True, scale=True,
        name=name, fused=True)(inputs)


def conv2d_layers(inputs, is_training, num_conv, data_format):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor (num_inputs, size, size, depth)
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        Conv features (num_inputs, num_dense).
    """
    convolved = inputs
    for i, num_filters in enumerate(num_conv):
        convolved_input = convolved
        # Add batch norm layer if enabled
        convolved_input = batch_norm(
            inputs=convolved_input, data_format=data_format,
            name='norm_conv2d_%d' % i)
        # Add dropout layer if enabled and not first conv layer
        if i > 0:
            convolved_input = tf.keras.layers.Dropout(
                rate=0.5, name='drop_conv2d_%d' % i)(convolved_input)
        # Add 2d convolution layer
        convolved = tf.keras.layers.Conv2D(
            filters=num_filters, kernel_size=3, padding='same',
            data_format=data_format, activation=tf.nn.relu,
            name='conv2d_%d' % i)(convolved_input)
        # Add pooling layer
        convolved = tf.keras.layers.MaxPool2D(
            pool_size=2, strides=2, data_format=data_format,
            name='pool_conv2d_%d' % i)(convolved)
    return convolved


class Model(object):
    """Base class for building ConvLSTM network."""

    def __init__(self, params):
        """Create a model to learn features on an object of the dimensions
            [seq_length, width, depth, channels].

        Args:
            params: Hyperparameters.
        """
        self.num_conv = [32, 32, 64, 64]
        self.frame_size = 128
        self.seq_length = 16
        self.num_dense = 1024
        self.num_lstm = 128
        self.num_classes = 2
        self.data_format = params.data_format

    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to learn features on a batch of image sequences.

        Args:
            inputs: A tensor representing a batch of input image sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, seq_length, num_classes + 1]
        """
        with tf.variable_scope(scope):
            # Reshape and feed all images through CNN
            num_channels = inputs.get_shape()[4]
            inputs = tf.reshape(inputs,
                [-1, self.frame_size, self.frame_size, num_channels])
            # Convert to channels_first if necessary (performance boost)
            if self.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
            inputs = conv2d_layers(
                inputs=inputs, is_training=is_training,
                num_conv=self.num_conv, data_format=self.data_format)
            inputs = tf.keras.layers.Flatten()(inputs)
            inputs = tf.keras.layers.Dropout(
                rate=0.5, name='drop_dense_0')(inputs)
            inputs = tf.keras.layers.Dense(
                units=self.num_dense, activation=tf.nn.relu,
                name='dense_0')(inputs)
            # Reshape and feed through sequence-aware LSTM
            inputs = tf.reshape(inputs, [-1, self.seq_length, self.num_dense])
            inputs = tf.keras.layers.LSTM(
                units=self.num_lstm, return_sequences=True,
                name='lstm_0')(inputs)
            # Classification layer
            inputs = tf.keras.layers.Dropout(
                rate=0.5, name='drop_dense_1')(inputs)
            inputs = tf.keras.layers.Dense(
                units=self.num_classes + 1, name='dense_1')(inputs)

        return inputs
