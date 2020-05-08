"""1D ResNet-18 CNN-LSTM Model for OREBA-DIS dataset"""

import tensorflow as tf

SEQ_POOL = 8
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class Conv1D(tf.keras.layers.Layer):
    """Strided 1-d convolution with explicit padding"""

    def __init__(self, filters, kernel_size, l2_lambda):
        super(Conv1D, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=kernel_size,
            strides=1, use_bias=True, padding='same',
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def call(self, inputs):
        inputs = self.conv1d(inputs)
        return inputs


class ConvBlock(tf.keras.layers.Layer):
    """One residual conv block"""

    def __init__(self, num_filters, kernel_size, shortcut, pool, l2_lambda):
        super(ConvBlock, self).__init__()
        self.shortcut = shortcut
        self.pool = pool
        self.conv_1 = Conv1D(filters=num_filters,
            kernel_size=kernel_size, l2_lambda=l2_lambda)
        self.conv_2 = Conv1D(filters=num_filters,
            kernel_size=kernel_size, l2_lambda=l2_lambda)
        if self.shortcut:
            self.conv_sc = Conv1D(filters=num_filters,
                kernel_size=1, l2_lambda=l2_lambda)
        self.relu = tf.keras.layers.ReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        self.bn_2 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        if self.pool:
            self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)

    @tf.function
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.shortcut:
            shortcut = self.conv_sc(shortcut)
        inputs = self.conv_1(inputs)
        inputs = self.relu(inputs)
        inputs = self.bn_1(inputs)
        inputs = self.conv_2(inputs)
        inputs = tf.keras.layers.add([inputs, shortcut])
        inputs = self.relu(inputs)
        inputs = self.bn_2(inputs)
        if self.pool:
            inputs = self.max_pool(inputs)
        return inputs


class LSTMBlock(tf.keras.layers.Layer):
    """One LSTM layer with residual connection"""

    def __init__(self, num_units, shortcut, l2_lambda):
        super(LSTMBlock, self).__init__()
        self.num_units = num_units
        self.shortcut = shortcut
        self.lstm = tf.keras.layers.LSTM(
            units=num_units, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        if self.shortcut:
            self.conv = Conv1D(filters=num_units,
                kernel_size=1, l2_lambda=l2_lambda)

    @tf.function
    def call(self, inputs, training=None):
        if self.shortcut:
            # If num_units == number of input features: direct shortcut
            if self.num_units == inputs.shape[2]:
                shortcut = inputs
            # If not equal: Use conv layer to learn same amount of features
            else:
                shortcut = self.conv(inputs)

        inputs = self.lstm(inputs)
        if self.shortcut:
            inputs = tf.keras.layers.add([inputs, shortcut])
        return inputs


class Model(tf.keras.Model):
    """Residual CNN-LSTM Model"""

    def __init__(self, num_classes, input_length, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.block_specs = [(128, 7, False), (128, 7, True), (256, 5, True),
          (265, 3, True)]
        self.lstm_specs = [(64, False), (64, True)]
        self.conv_1 = Conv1D(filters=64, kernel_size=7, l2_lambda=l2_lambda)
        self.relu = tf.keras.layers.ReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        self.conv_blocks = []
        for i, (filters, kernel_size, pool) in enumerate(self.block_specs):
            self.conv_blocks.append(ConvBlock(
                num_filters=filters, kernel_size=kernel_size, shortcut=True,
                pool=pool, l2_lambda=l2_lambda))
        self.lstm_blocks = []
        for i, (num_units, shortcut) in enumerate(self.lstm_specs):
            self.lstm_blocks.append(LSTMBlock(
                num_units=num_units, shortcut=shortcut, l2_lambda=l2_lambda))
        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def call(self, inputs, training=False):
        inputs = self.conv_1(inputs)
        inputs = self.relu(inputs)
        inputs = self.bn_1(inputs)
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs, training=training)
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs, training=training)
        inputs = self.dense(inputs)
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
