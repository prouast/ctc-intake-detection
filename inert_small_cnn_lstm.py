"""CNN-LSTM Model"""

import tensorflow as tf

SCOPE = "inert_small_cnn_lstm"

def batch_norm(inputs, name):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=2, center=True, scale=True, name=name, fused=True)(inputs)

class Model(object):
    """Base class for building CNN-LSTM network."""

    def __init__(self, params):
        """Create a model to learn features on an object of the dimensions
            [seq_length, channels].

        Args:
            params: Hyperparameters.
        """
        self.num_conv = [64, 64, 128, 128]
        self.num_channels = 12
        self.seq_length = params.seq_length
        self.num_dense = 32
        self.num_lstm = [64, 64]
        self.num_classes = params.num_classes

    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to learn features on a sequence of inertial measurements.

        Args:
            inputs: A [batch_size, seq_length, 12] tensor.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, seq_length//4, num_classes + 1]
        """
        with tf.variable_scope(scope):
            # Conv layers
            for i, num_filters in enumerate(self.num_conv):
                inputs = tf.keras.layers.BatchNormalization(
                    axis=2, center=True, scale=True, name='norm_conv2d_%d' % i,
                    fused=True)(inputs)
                if i > 0:
                    inputs = tf.keras.layers.Dropout(
                        rate=0.5, name='drop_conv2d_%d' % i)(inputs)
                inputs = tf.keras.layers.Conv1D(
                    filters=num_filters, kernel_size=7, padding='same',
                    activation=tf.nn.relu, name='conv2d_%d' % i)(inputs)
                if i % 2 == 0:
                    inputs = tf.keras.layers.MaxPool1D(
                        pool_size=2, strides=2, name='pool_conv2d_%d' % i)(inputs)
            # Dense layer
            inputs = tf.keras.layers.Dropout(
                rate=0.5, name='drop_dense_0')(inputs)
            inputs = tf.keras.layers.Dense(
                units=self.num_dense, activation=tf.nn.relu,
                name='dense_0')(inputs)
            # LSTM layers
            for i, num_units in enumerate(self.num_lstm):
                inputs = tf.keras.layers.LSTM(
                    units=num_units, return_sequences=True,
                    name='lstm_%d' % i)(inputs)
            # Classification layer
            inputs = tf.keras.layers.Dropout(
                rate=0.5, name='drop_dense_1')(inputs)
            inputs = tf.keras.layers.Dense(
                units=self.num_classes + 1, name='dense_1')(inputs)

        return inputs
