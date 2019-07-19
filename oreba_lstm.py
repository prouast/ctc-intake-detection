"""Simple LSTM Model"""

import tensorflow as tf

SCOPE = "oreba_lstm"


class Model(object):
    """Base class for building LSTM network."""

    def __init__(self, params):
        """Create a model to learn features on an object of the dimensions
            [seq_length, fc7 features].

        Args:
            params: Hyperparameters.
        """
        self.num_lstm = 128
        self.num_classes = 2
        self.data_format = params.data_format

    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to learn features on a batch of sequences of fc7 features.

        Args:
            inputs: A tensor representing a batch of fc7 feature sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, seq_length, num_classes + 1]
        """
        with tf.variable_scope(scope):
            inputs = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.num_lstm,
                    dropout=0.5,
                    recurrent_dropout=0.5,
                    return_sequences=True))(inputs)
            inputs = tf.keras.layers.Dense(
                units=self.num_classes + 1)(inputs)
        return inputs
