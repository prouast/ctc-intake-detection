"""CNN-LSTM Model"""

import tensorflow as tf

class Model(tf.keras.Model):
    """Base class for building ConvLSTM network."""

    def __init__(self, seq_length, num_classes, l2_lambda):
        """Create a model to learn features on an object of the dimensions
            [seq_length, width, depth, channels].

        Args:
            params: Hyperparameters.
        """
        self.l2_lambda = l2_lambda
        self.num_conv = [32, 32, 64, 64]
        self.frame_size = 128
        self.num_channels = 3
        self.seq_length = seq_length
        self.num_dense = 1024
        self.num_lstm = 128
        self.num_classes = num_classes

    def __call__(self, inputs, is_training):
        """Add operations to learn features on a batch of image sequences.

        Args:
            inputs: A tensor representing a batch of input image sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, seq_length, num_classes + 1]
        """
        # Reshape for CNN
        inputs = tf.reshape(inputs,
            [-1, self.frame_size, self.frame_size, self.num_channels])
        # Conv blocks
        for i, num_filters in enumerate(self.num_conv):
            # Batch norm
            inputs = tf.keras.layers.BatchNormalization(
                axis=3, center=True, scale=True, fused=True)(inputs)
            # Dropout if not first conv layer
            if i > 0:
                inputs = tf.keras.layers.Dropout(
                    rate=0.5)(inputs)
            # 2d conv
            inputs = tf.keras.layers.Conv2D(
                filters=num_filters, kernel_size=3, padding='same',
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)
            # Max pool
            inputs = tf.keras.layers.MaxPool2D(
                pool_size=2, strides=2)(inputs)
        # Dense
        inputs = tf.keras.layers.Flatten()(inputs)
        inputs = tf.keras.layers.Dropout(
            rate=0.5)(inputs)
        inputs = tf.keras.layers.Dense(
            units=self.num_dense, activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)
        # Reshape for LSTM
        inputs = tf.reshape(inputs, [-1, self.seq_length, self.num_dense])
        inputs = tf.keras.layers.LSTM(
            units=self.num_lstm, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)
        # Classification
        inputs = tf.keras.layers.Dropout(
            rate=0.5)(inputs)
        inputs = tf.keras.layers.Dense(
            units=self.num_classes + 1,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda))(inputs)

        return inputs
