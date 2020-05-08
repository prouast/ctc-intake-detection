"""ResNet-50 CNN-LSTM Model for OREBA-DIS dataset from Rouast et al. (2019)"""

import tensorflow as tf

SEQ_POOL = 1
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
IMAGENET_SIZE = 224


class Conv2DFixedPadding(tf.keras.layers.Layer):
    """Strided 2-d convolution with explicit padding"""

    def __init__(self, filters, kernel_size, strides):
        super(Conv2DFixedPadding, self).__init__()
        self.strides_one = strides == 1
        self.kernel_size = kernel_size
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides), use_bias=False,
            padding=('same' if self.strides_one else 'valid'),
            kernel_initializer=tf.keras.initializers.VarianceScaling())

    @tf.function
    def call(self, inputs):
        if not self.strides_one:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(tensor=inputs,
                paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        inputs = self.conv2d(inputs)
        return inputs


class BottleneckResBlock(tf.keras.layers.Layer):
    """A single block for ResNet v2 with bottleneck"""

    def __init__(self, filters, shortcut, strides):
        super(Block, self).__init__()
        self.shortcut = shortcut
        self.conv_1 = Conv2DFixedPadding(
            filters=filters, kernel_size=1, strides=1)
        self.bn_1 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
            center=True, scale=True, fused=True)
        self.conv_2 = Conv2DFixedPadding(
            filters=filters, kernel_size=3, strides=strides)
        self.bn_2 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
            center=True, scale=True, fused=True)
        self.conv_3 = Conv2DFixedPadding(
            filters=4*filters, kernel_size=1, strides=1)
        self.bn_3 = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
            center=True, scale=True, fused=True)
        if self.shortcut:
            self.conv_sc = Conv2DFixedPadding(
                filters=4*filters, kernel_size=1, strides=strides)
            self.bn_sc = tf.keras.layers.BatchNormalization(
                momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                center=True, scale=True, fused=True)

    @tf.function
    def call(self, inputs):
        shortcut = inputs
        if self.shortcut:
            shortcut = self.conv_sc(inputs)
            shortcut = self.bn_sc(inputs)
        inputs = self.conv_1(inputs)
        inputs = self.bn_1(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv_2(inputs)
        inputs = self.bn_2(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv_3(inputs)
        inputs = self.bn_3(inputs)
        inputs = tf.keras.layers.add([inputs, shortcut])
        inputs = self.relu(inputs)
        return inputs


class BlockLayer(tf.keras.layers.Layer):
    """One layer of blocks for a ResNet model"""

    def __init__(self, filters, blocks, strides, name):
        super(BlockLayer, self).__init__()
        self.name_ = name
        self.block = BottleneckResBlock(
            filters=filters, shortcut=True, strides=strides)
        self.blocks = []
        for i in range(blocks-1):
            self.blocks.append(
                BottleneckResBlock(filters=filters, shortcut=False, strides=1))

    @tf.function
    def call(self, inputs):
        inputs = self.block(inputs)
        for block in self.blocks:
            inputs = block(inputs)
        return tf.identity(inputs, self.name_)


class Model(tf.keras.Model):
    """ResNet-50 CNN-LSTM Model"""

    def __init__(self, num_classes, input_length, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.block_sizes = [3, 4, 6, 3]
        self.block_strides = [1, 2, 2, 2]
        self.num_filters = 64
        self.kernel_size = 7
        self.num_lstm = 128
        self.conv_1 = tf.keras.layers.TimeDistributed(
            Conv2DFixedPadding(
                filters=self.num_filters, kernel_size=7, strides=2))
        self.pool_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool2D(
                pool_size=(3, 3), strides=(2, 2), padding='same'))
        self.bn_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization(
                momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                center=True, scale=True, fused=True))
        self.relu = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())
        self.block_layers = []
        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2**i)
            num_strides = self.block_strides[i]
            self.block_layers.append(
                tf.keras.layers.TimeDistributed(
                    BlockLayer(
                        filters=num_filters, blocks=num_blocks,
                        strides=num_strides,
                        name='block_layer_{}'.format(i+1))))
        self.lstm = tf.keras.layers.LSTM(
            units=self.num_lstm, return_sequences=True)
        self.dense = tf.keras.layers.Dense(
            units=num_classes)

    @tf.function
    def call(self, inputs, training=False):
        # Resize inputs to ImageNet size
        num_seq = inputs.get_shape()[1]
        original_frame_size = inputs.get_shape()[2]
        num_channels = inputs.get_shape()[4]
        inputs = tf.reshape(inputs,
            [-1, original_frame_size, original_frame_size, num_channels])
        inputs = tf.image.resize(inputs, [IMAGENET_SIZE, IMAGENET_SIZE])
        inputs = tf.reshape(inputs,
            [-1, num_seq, IMAGENET_SIZE, IMAGENET_SIZE, num_channels])
        # ResNet-50
        inputs = self.conv_1(inputs)
        inputs = self.pool_1(inputs)
        inputs = self.bn_1(inputs)
        inputs = self.relu(inputs)
        for block_layer in self.block_layers:
            inputs = block_layer(inputs)
        # Average pooling
        inputs = tf.reduce_mean(input_tensor=inputs, axis=[2, 3], keepdims=True)
        inputs = tf.identity(inputs, 'average_pool')
        inputs = tf.squeeze(inputs, [2, 3])
        # LSTM
        inputs = self.lstm(inputs)
        inputs = self.dense(inputs)
        return inputs

    @tf.function
    def labels(self, labels, batch_size=None):
        """No pooling"""
        return labels

    def seq_length(self):
        return self.input_length

    def seq_pool(self):
        return SEQ_POOL
