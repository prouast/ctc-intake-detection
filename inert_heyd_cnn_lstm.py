"""CNN-LSTM Model for inertial data from Heydarian et al. (2020)"""

import tensorflow as tf


class Model(tf.keras.Model):
    """CNN-LSTM for inertial data"""

    def __init__(self, num_classes, input_length, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.conv1d_1 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=1, padding='valid',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, padding='valid',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_3 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=5, padding='valid',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_4 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=7, padding='valid',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.lstm_1 = tf.keras.layers.LSTM(
            units=64, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.lstm_2 = tf.keras.layers.LSTM(
            units=64, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dense_1 = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def call(self, inputs, training=False):
        inputs = self.conv1d_1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.conv1d_2(inputs)
        inputs = self.dropout(inputs)
        inputs = self.conv1d_3(inputs)
        inputs = self.dropout(inputs)
        inputs = self.conv1d_4(inputs)
        inputs = self.dropout(inputs)
        inputs = self.lstm_1(inputs)
        inputs = self.lstm_2(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dropout(inputs)
        return inputs

    def get_label_fn(self, batch_size=None):
        """Returns the function needed for adjusting label dims"""
        @tf.function
        def labels_with_batch_dim(labels):
            """Truncate according to convolutions"""
            return tf.slice(labels, [0, 6], [batch_size, self.input_length-12])
        @tf.function
        def labels_without_batch_dim(labels):
            """Truncate according to convolutions"""
            return tf.slice(labels, [6], [self.input_length-12])

        if batch_size is not None:
            return labels_with_batch_dim
        else:
            return labels_without_batch_dim

    def get_seq_length(self):
        return self.input_length-12

    def get_seq_pool(self):
        return SEQ_POOL
