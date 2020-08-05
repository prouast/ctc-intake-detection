"""ResNet-50 CNN-LSTM Model for video data from Rouast et al. (2019)"""

import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class Conv2DFixedPadding(tf.keras.layers.Layer):
  """Strided 2-d convolution with explicit padding"""

  def __init__(self, filters, kernel_size, strides, l2_lambda):
    super(Conv2DFixedPadding, self).__init__()
    self.strides_one = strides == 1
    self.kernel_size = kernel_size
    self.conv2d = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=(kernel_size, kernel_size),
      strides=(strides, strides), use_bias=False,
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
        paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    inputs = self.conv2d(inputs)
    return inputs


class BottleneckResBlock(tf.keras.layers.Layer):
  """A single block for ResNet v2 with bottleneck"""

  def __init__(self, filters, shortcut, strides, l2_lambda):
    super(BottleneckResBlock, self).__init__()
    self.shortcut = shortcut
    self.conv_1 = Conv2DFixedPadding(
      filters=filters, kernel_size=1, strides=1, l2_lambda=l2_lambda)
    self.bn_1 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.conv_2 = Conv2DFixedPadding(
      filters=filters, kernel_size=3, strides=strides, l2_lambda=l2_lambda)
    self.bn_2 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.conv_3 = Conv2DFixedPadding(
      filters=4*filters, kernel_size=1, strides=1, l2_lambda=l2_lambda)
    self.bn_3 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.relu = tf.keras.layers.ReLU()
    if self.shortcut:
      self.conv_sc = Conv2DFixedPadding(
        filters=4*filters, kernel_size=1, strides=strides, l2_lambda=l2_lambda)
      self.bn_sc = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)

  @tf.function
  def call(self, inputs):
    shortcut = inputs
    if self.shortcut:
      shortcut = self.conv_sc(shortcut)
      shortcut = self.bn_sc(shortcut)
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

  def __init__(self, filters, blocks, strides, l2_lambda):
    super(BlockLayer, self).__init__()
    self.block = BottleneckResBlock(
      filters=filters, shortcut=True, strides=strides, l2_lambda=l2_lambda)
    self.blocks = []
    for i in range(blocks-1):
      self.blocks.append(
        BottleneckResBlock(filters=filters, shortcut=False, strides=1, l2_lambda=l2_lambda))

  @tf.function
  def call(self, inputs):
    inputs = self.block(inputs)
    for block in self.blocks:
      inputs = block(inputs)
    return inputs


class LSTMLayer(tf.keras.layers.Layer):
  """One LSTM layer with residual connection"""

  def __init__(self, num_units, l2_lambda):
    super(LSTMLayer, self).__init__()
    self.num_units = num_units
    self.lstm = tf.keras.layers.LSTM(
      units=num_units, return_sequences=True,
      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

  @tf.function
  def call(self, inputs, training=None):
    inputs = self.lstm(inputs)
    return inputs


class Model(tf.keras.Model):
  """ResNet-50 CNN-LSTM Model"""

  def __init__(self, num_classes, input_length, l2_lambda):
    super(Model, self).__init__()
    self.input_length = input_length
    self.block_sizes = [3, 4, 6, 3]
    self.block_strides = [1, 2, 2, 2]
    self.num_filters = 64
    self.lstm_specs = [128]
    self.conv_1 = tf.keras.layers.TimeDistributed(
      Conv2DFixedPadding(
        filters=self.num_filters, kernel_size=5, strides=1, l2_lambda=l2_lambda))
    self.pool_1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='same'))
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
            strides=num_strides, l2_lambda=l2_lambda)))
    self.lstm_blocks = []
    for i, num_units in enumerate(self.lstm_specs):
      self.lstm_blocks.append(LSTMLayer(
        num_units=num_units, l2_lambda=l2_lambda))
    self.dense = tf.keras.layers.Dense(
      units=num_classes)
    self.linear = tf.keras.layers.Activation("linear", dtype="float32")

  @tf.function
  def call(self, inputs, training=False):
    inputs = self.conv_1(inputs)
    inputs = self.pool_1(inputs)
    inputs = self.bn_1(inputs)
    inputs = self.relu(inputs)
    for block_layer in self.block_layers:
      inputs = block_layer(inputs)
    inputs = tf.reduce_mean(input_tensor=inputs, axis=[2, 3], keepdims=True)
    inputs = tf.identity(inputs, 'average_pool')
    inputs = tf.squeeze(inputs, [2, 3])
    for lstm_block in self.lstm_blocks:
      inputs = lstm_block(inputs, training=training)
    inputs = self.dense(inputs)
    inputs = self.linear(inputs)
    return inputs

  def get_label_fn(self, batch_size=None):
    """Returns the function needed for adjusting label dims"""
    def labels(labels):
      """No pooling"""
      return labels
    return labels

  def get_seq_length(self):
    return self.input_length

  def get_seq_pool(self):
    return 1
