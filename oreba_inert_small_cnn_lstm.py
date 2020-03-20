"""CNN-LSTM Model for OREBA-DIS dataset"""

import tensorflow as tf

class ConvBlock(tf.keras.Model):
    """One block of Conv1D-BN-Dropout-MaxPool1D"""

    def __init__(self, num_filters, max_pool, l2_lambda):
        super(ConvBlock, self).__init__()
        self.max_pool = max_pool
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters, kernel_size=7, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        if max_pool:
            self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)

    @tf.function
    def __call__(self, inputs, training=False):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = self.dropout(inputs)
        if self.max_pool:
            inputs = self.max_pool(inputs)
        return inputs

class Model(tf.keras.Model):
    """CNN-LSTM Model for inertial data"""

    def __init__(self, num_classes, seq_pool, l2_lambda):
        super(Model, self).__init__()
        # Make sure model implied seq_pool equals arg implied seq_pool
        assert seq_pool == 8, \
            "seq_pool: Model implied == 8 != {} == arg implied".format(seq_pool)
        self.num_conv = [64, 128, 256]
        self.num_lstm = [64, 128]
        self.conv_blocks = []
        for i, num_filters in enumerate(self.num_conv):
            self.conv_blocks.append(ConvBlock(num_filters, True, l2_lambda))
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
    def __call__(self, inputs, training=False):
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs)
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs)
        inputs = self.dense(inputs)
        inputs = self.dropout(inputs)
        return inputs
