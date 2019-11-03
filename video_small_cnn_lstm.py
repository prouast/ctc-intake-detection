"""CNN-LSTM Model"""

import tensorflow as tf

class ConvBlock(tf.keras.Model):
    """One block of Conv2D-BN-Dropout-MaxPool2D"""
    def __init__(self, num_filters, l2_lambda):
        super(ConvBlock, self).__init__()
        self.max_pool = max_pool
        self.conv = tf.keras.layers.Conv2D(
            filters=num_filters, kernel_size=3, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
    def __call__(self, inputs, training=False):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = self.dropout(inputs)
        inputs = self.max_pool(inputs)
        return inputs

class Model(tf.keras.Model):
    """CNN-LSTM Model for inertial data"""

    def __init__(self, num_classes, l2_lambda):
        super(Model, self).__init__()
        self.num_conv = [32, 32, 64, 64]
        self.num_dense = 1024
        self.num_lstm = [64, 128]
        self.conv_blocks = []
        for i, num_filters in enumerate(self.num_conv):
            self.conv_blocks.append(ConvBlock(num_filters, l2_lambda))
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(
            units=self.num_dense, activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.lstm_blocks = []
        for i, num_units in enumerate(self.num_lstm):
            self.lstm_blocks.append(tf.keras.layers.LSTM(
                units=num_units, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
        self.dense_2 = tf.keras.layers.Dense(
            units=num_classes + 1,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    def __call__(self, inputs, training=False):
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs)
        inputs = self.flatten(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dropout(inputs)
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs)
        inputs = self.dense_2(inputs)
        inputs = self.dropout(inputs)
        return inputs
