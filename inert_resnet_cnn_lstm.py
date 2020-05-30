"""1D ResNet-18 CNN-LSTM Model for inertial data"""

import tensorflow as tf


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
    def call(self, inputs):
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

    @tf.function
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.shortcut:
            shortcut = self.conv_sc(shortcut)
        inputs = self.conv_1(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv_2(inputs)
        inputs = tf.keras.layers.add([inputs, shortcut])
        inputs = self.relu(inputs)
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
    def call(self, inputs, training=None):
        inputs = self.block(inputs, training=training)
        for block in self.blocks:
            inputs = block(inputs, training=training)
        return inputs


class LSTMLayer(tf.keras.layers.Layer):
    """One LSTM layer with residual connection"""

    def __init__(self, num_units, shortcut, l2_lambda):
        super(LSTMLayer, self).__init__()
        self.num_units = num_units
        self.shortcut = shortcut
        self.lstm = tf.keras.layers.LSTM(
            units=num_units, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        if self.shortcut:
            self.conv = Conv1DFixedPadding(filters=num_units,
                kernel_size=1, strides=1, l2_lambda=l2_lambda)

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

    def __init__(self, num_classes, input_length, specs, l2_lambda):
        super(Model, self).__init__()
        self.input_length = input_length
        self.seq_pool = specs["seq_pool"]
        self.block_specs = specs["block_specs"]
        self.lstm_specs = specs["lstm_specs"]
        self.conv_1 = Conv1DFixedPadding(filters=specs["conv_1_filters"],
            kernel_size=specs["conv_1_kernel_size"], strides=1,
            l2_lambda=l2_lambda)
        self.relu = tf.keras.layers.ReLU()
        #self.bn_1 = tf.keras.layers.BatchNormalization(
        #    momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
        self.conv_blocks = []
        for i, (blocks, filters, kernel_size, strides) in enumerate(self.block_specs):
            self.conv_blocks.append(BlockLayer(
                num_blocks=blocks, num_filters=filters,
                kernel_size=kernel_size, strides=strides, l2_lambda=l2_lambda))
        self.lstm_blocks = []
        for i, (num_units, shortcut) in enumerate(self.lstm_specs):
            self.lstm_blocks.append(LSTMLayer(
                num_units=num_units, shortcut=shortcut, l2_lambda=l2_lambda))
        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def call(self, inputs, training=False):
        inputs = self.conv_1(inputs)
        inputs = self.relu(inputs)
        #inputs = self.bn_1(inputs)
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs, training=training)
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs, training=training)
        inputs = self.dense(inputs)
        return inputs

    def get_label_fn(self, batch_size=None):
        """Returns the function needed for adjusting label dims"""
        @tf.function
        def labels_with_batch_dim(labels):
            """Slice labels corresponding to pooling layers in model"""
            labels = tf.strided_slice(
                input_=labels, begin=[0, self.seq_pool-1],
                end=[batch_size, self.input_length],
                strides=[1, self.seq_pool])
            labels = tf.reshape(labels,
                [batch_size, int(self.input_length/self.seq_pool)])
            return labels
        @tf.function
        def labels_without_batch_dim(labels):
            labels = tf.strided_slice(
                input_=labels, begin=[self.seq_pool-1], end=[self.input_length],
                strides=[self.seq_pool])
            labels = tf.reshape(labels, [int(self.input_length/self.seq_pool)])
            return labels

        if batch_size is not None:
            return labels_with_batch_dim
        else:
            return labels_without_batch_dim

    def get_seq_length(self):
        return int(self.input_length/self.seq_pool)

    def get_seq_pool(self):
        return self.seq_pool
