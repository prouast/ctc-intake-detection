"""Simple LSTM Model"""

import tensorflow as tf

class Model(tf.keras.Model):
    """Base class for building LSTM network."""

    def __init__(self, num_classes, l2_lambda):
        """Create a model to learn features on an object of the dimensions
            [seq_length, fc7 features].

        Args:
            params: Hyperparameters.
        """
        self.l2_lambda = l2_lambda
        self.num_lstm = 128
        self.num_classes = num_classes

    def __call__(self, inputs, is_training):
        """Add operations to learn features on a batch of sequences of fc7 features.

        Args:
            inputs: A tensor representing a batch of fc7 feature sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, seq_length, num_classes + 1]
        """
        inputs = tf.keras.layers.LSTM(
            units=self.num_lstm, dropout=0.5, recurrent_dropout=0.5,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)
        inputs = tf.keras.layers.Dense(
            units=self.num_classes + 1,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)

        return inputs
