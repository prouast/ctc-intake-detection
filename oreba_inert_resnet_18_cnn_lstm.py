"""1D ResNet-18 CNN-LSTM Model for OREBA-DIS dataset"""

import tensorflow as tf

SEQ_POOL = 8
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class Conv1DFixedPadding(tf.keras.layers.Layer):
    """Strided 1-d convolution with explicit padding"""

    def __init__(self, filters, kernel_size, strides, l2_lambda):
        super(Conv1DFixedPadding, self).__init__()
        self.strides_one = strides == 1
        self.kernel_size = kernel_size
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, use_bias=True,
            padding=('same' if self.strides_one else 'valid'),
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def __call__(self, inputs):
        if not self.strides_one:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(tensor=inputs,
                paddings=[[0, 0], [pad_beg, pad_end], [0, 0]])
        inputs = self.conv1d(inputs)
        return inputs


class ResBlock(tf.keras.layers.Layer):
    """One residual block"""

    def __init__(self, num_filters, kernel_size, shortcut, strides, l2_lambda):
        super(ResBlock, self).__init__()
        self.shortcut = shortcut
        self.conv_1 = Conv1DFixedPadding(filters=num_filters,
            kernel_size=kernel_size, strides=strides, l2_lambda=l2_lambda)
        self.conv_2 = Conv1DFixedPadding(filters=num_filters,
            kernel_size=kernel_size, strides=1, l2_lambda=l2_lambda)
        if self.shortcut:
            self.conv_sc = Conv1DFixedPadding(filters=num_filters,
                kernel_size=1, strides=strides, l2_lambda=l2_lambda)
        self.relu = tf.keras.layers.ReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        self.bn_2 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)

    @tf.function
    def __call__(self, inputs, training=None):
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
        return inputs


class BlockLayer(tf.keras.layers.Layer):
    """One layer of blocks"""

    def __init__(self, num_blocks, num_filters, kernel_size, strides, l2_lambda):
        super(BlockLayer, self).__init__()
        self.block = ResBlock(
            num_filters=num_filters, kernel_size=kernel_size, shortcut=True,
            strides=strides, l2_lambda=l2_lambda)
        self.blocks = []
        for i in range(num_blocks-1):
            self.blocks.append(
                ResBlock(
                    num_filters=num_filters, kernel_size=kernel_size,
                    shortcut=False, strides=1, l2_lambda=l2_lambda))

    @tf.function
    def __call__(self, inputs, training=None):
        inputs = self.block(inputs, training=training)
        for block in self.blocks:
            inputs = block(inputs, training=training)
        return inputs


class Model(tf.keras.Model):
    """Residual CNN-LSTM Model"""

    def __init__(self, num_classes, input_length, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.block_specs = [(2, 64, 5, 1), (2, 128, 5, 2), (2, 256, 3, 2),
            (2, 512, 3, 2)]
        self.lstm_specs = [64, 64]
        self.conv_1 = Conv1DFixedPadding(filters=64,
            kernel_size=7, strides=1, l2_lambda=l2_lambda)
        self.relu = tf.keras.layers.ReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        self.conv_blocks = []
        for i, (blocks, filters, kernel_size, strides) in enumerate(self.block_specs):
            self.conv_blocks.append(BlockLayer(
                num_blocks=blocks, num_filters=filters,
                kernel_size=kernel_size, strides=strides, l2_lambda=l2_lambda))
        self.lstm_blocks = []
        for i, num_units in enumerate(self.lstm_specs):
            self.lstm_blocks.append(tf.keras.layers.LSTM(
                units=num_units, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def __call__(self, inputs, training=False):
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