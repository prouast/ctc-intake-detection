"""CNN-LSTM Model for OREBA-DIS dataset"""

import tensorflow as tf

SEQ_POOL = 16


class ConvBlock(tf.keras.layers.Layer):
    """One block of Conv1D-BN-Dropout-MaxPool1D"""

    def __init__(self, num_filters, kernel_size, max_pool, l2_lambda):
        super(ConvBlock, self).__init__()
        self.max_pool = max_pool
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters, kernel_size=kernel_size, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        if max_pool:
            self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)

    @tf.function
    def call(self, inputs, training):
        inputs = self.conv(inputs)
        inputs = self.relu(inputs)
        inputs = self.dropout(inputs)
        if self.max_pool:
            inputs = self.max_pool(inputs)
        return inputs

class Model(tf.keras.Model):
    """CNN-LSTM Model for inertial data"""

    def __init__(self, num_classes, input_length, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.num_conv = [(64, 7, True), (128, 5, True), (256, 3, True)]
        self.num_lstm = [64]
        self.conv_blocks = []
        for i, (num_filters, kernel_size, max_pool) in enumerate(self.num_conv):
            self.conv_blocks.append(ConvBlock(num_filters=num_filters,
                kernel_size=kernel_size, max_pool=max_pool, l2_lambda=l2_lambda))
        self.lstm_blocks = []
        for i, num_units in enumerate(self.num_lstm):
            self.lstm_blocks.append(tf.keras.layers.LSTM(
                units=num_units, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    @tf.function
    def call(self, inputs, training):
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs, training=training)
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs)
        inputs = self.dense(inputs)
        inputs = self.dropout(inputs)
        return inputs

    @tf.function
    def labels(self, labels, batch_size=None):
        """Slice labels corresponding to pooling layers in model"""
        if batch_size is not None:
            labels = tf.strided_slice(
                input_=labels, begin=[0, SEQ_POOL-1], end=[batch_size, self.input_length],
                strides=[1, SEQ_POOL])
            labels = tf.reshape(labels, [batch_size, int(self.input_length/SEQ_POOL)])
        else:
            labels = tf.strided_slice(
                input_=labels, begin=[SEQ_POOL-1], end=[self.input_length],
                strides=[SEQ_POOL])
            labels = tf.reshape(labels, [int(self.input_length/SEQ_POOL)])
        return labels

    def seq_length(self):
        return int(self.input_length / SEQ_POOL)

    def seq_pool(self):
        return SEQ_POOL
