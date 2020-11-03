"""ResNet-50 SlowFast Model for video data from Rouast et al. (2019)"""

import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class Conv3DFixedPadding(tf.keras.layers.Layer):
  """Strided 3-d convolution with explicit padding"""

  def __init__(self, filters, kernel_size, strides, l2_lambda):
    super(Conv3DFixedPadding, self).__init__()
    if (isinstance(strides, list) and max(strides) > 1) or \
       (isinstance(strides, int) and strides > 1):
      self.strides_one = False
    else:
      self.strides_one = True
    self.padding_kernel_size = max(kernel_size) if isinstance(kernel_size, list) else kernel_size
    self.kernel_size = kernel_size
    self.conv3d = tf.keras.layers.Conv3D(
      filters=filters, kernel_size=kernel_size,
      strides=strides, use_bias=False,
      padding=('same' if self.strides_one else 'valid'),
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

  @tf.function
  def call(self, inputs):
    if not self.strides_one:
      pad_total = self.padding_kernel_size - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      inputs = tf.pad(tensor=inputs,
        paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    inputs = self.conv3d(inputs)
    return inputs


class Conv3DBottleneckResBlock(tf.keras.layers.Layer):
  """A single block for 3D ResNet v2 with bottleneck"""

  def __init__(self, filters, shortcut, strides, temp_kernel_size, l2_lambda):
    super(Conv3DBottleneckResBlock, self).__init__()
    self.bn_0 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.shortcut = shortcut
    self.conv_1 = Conv3DFixedPadding(
      filters=filters, kernel_size=[temp_kernel_size, 1, 1], strides=1, l2_lambda=l2_lambda)
    self.bn_1 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.conv_2 = Conv3DFixedPadding(
      filters=filters, kernel_size=[1, 3, 3], strides=strides, l2_lambda=l2_lambda)
    self.bn_2 = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.conv_3 = Conv3DFixedPadding(
      filters=4*filters, kernel_size=1, strides=1, l2_lambda=l2_lambda)
    self.relu = tf.keras.layers.ReLU()
    if self.shortcut:
      self.conv_sc = Conv3DFixedPadding(
        filters=4*filters, kernel_size=1, strides=strides, l2_lambda=l2_lambda)

  @tf.function
  def call(self, inputs):
    shortcut = inputs
    if self.shortcut:
      shortcut = self.conv_sc(shortcut)
    inputs = self.bn_0(inputs)
    inputs = self.relu(inputs)
    inputs = self.conv_1(inputs)
    inputs = self.bn_1(inputs)
    inputs = self.relu(inputs)
    inputs = self.conv_2(inputs)
    inputs = self.bn_2(inputs)
    inputs = self.relu(inputs)
    inputs = self.conv_3(inputs)
    inputs = tf.keras.layers.add([inputs, shortcut])
    return tf.inputs


class Conv3DBlockLayer(tf.keras.layers.Layer):
  """One layer of blocks for a ResNet model"""

  def __init__(self, filters, blocks, strides, temp_kernel_size, l2_lambda):
    super(Conv3DBlockLayer, self).__init__()
    self.block = Conv3DBottleneckResBlock(
      filters=filters, shortcut=True, strides=strides, temp_kernel_size=temp_kernel_size, l2_lambda=l2_lambda)
    self.blocks = []
    for i in range(blocks-1):
      self.blocks.append(
        Conv3DBottleneckResBlock(filters=filters, shortcut=False, strides=1, temp_kernel_size=temp_kernel_size, l2_lambda=l2_lambda))

  @tf.function
  def call(self, inputs):
    inputs = self.block(inputs)
    for block in self.blocks:
      inputs = block(inputs)
    return inputs


class Model(tf.keras.Model):
  """ResNet-50 SlowFast Model"""

  def __init__(self, num_classes, input_length, l2_lambda):
    super(Model, self).__init__()
    self.num_classes = num_classes
    self.input_length = input_length
    self.slowfast_alpha = 4
    self.slowfast_beta = 0.25
    self.num_filters = 64
    self.block_specs = [[3, 64, 1, 1], [4, 128, 2, 1], [6, 256, 2, 3], [3, 512, 2, 3]]
    self.conv_1_slow = Conv3DFixedPadding(
      filters=self.num_filters, kernel_size=[1, 5, 5], strides=1, l2_lambda=l2_lambda)
    self.conv_1_fast = Conv3DFixedPadding(
      filters=int(self.num_filters * self.slowfast_beta), kernel_size=[3, 5, 5], strides=1, l2_lambda=l2_lambda)
    self.max_pool_3d = tf.keras.layers.MaxPool3D(
      pool_size=[1, 3, 3], strides=[1, 2, 2], padding='same')
    self.block_layers = []
    for i, (num_blocks, num_filters_slow, num_strides, temp_kernel_size_slow) in enumerate(self.block_specs):
      num_filters_fast = self.slowfast_beta * num_filters_slow
      self.block_layers.append([
        tf.keras.layers.Conv3D(
          filters=num_filters_slow,
          kernel_size=[int(self.slowfast_alpha), 1, 1],
          strides=[int(self.slowfast_alpha), 1, 1], padding='valid'),
        Conv3DBlockLayer(
          filters=num_filters_slow, blocks=num_blocks, strides=[1, num_strides, num_strides],
          temp_kernel_size=temp_kernel_size_slow, l2_lambda=l2_lambda),
        Conv3DBlockLayer(
          filters=num_filters_fast, blocks=num_blocks, strides=[1, num_strides, num_strides],
          temp_kernel_size=3, l2_lambda=l2_lambda)])
    self.bn_slow = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.bn_fast = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)
    self.dense = tf.keras.layers.Dense(units=self.num_classes)

  @tf.function
  def call(self, inputs, training=False):
    inputs_slow = tf.stack([inputs[:,3], inputs[:,7], inputs[:,11], inputs[:,15]], axis=1)
    inputs_fast = inputs
    inputs_slow = self.conv_1_slow(inputs_slow)
    inputs_fast = self.conv_1_fast(inputs_fast)
    inputs_slow = self.max_pool_3d(inputs_slow)
    inputs_fast = self.max_pool_3d(inputs_fast)
    for i, (lat_conn_layer, block_layer_slow, block_layer_fast) in enumerate(self.block_layers):
      lat_conn = lat_conn_layer(inputs_fast)
      inputs_slow = tf.concat([inputs_slow, lat_conn], axis=4)
      inputs_slow = block_layer_slow(inputs_slow)
      inputs_fast = block_layer_fast(inputs_fast)
    inputs_slow = self.bn_slow(inputs_slow)
    inputs_fast = self.bn_fast(inputs_fast)
    inputs_slow = tf.nn.relu(inputs_slow)
    inputs_fast = tf.nn.relu(inputs_fast)
    inputs_slow = tf.reduce_mean(inputs_slow, axis=[1, 2, 3], keepdims=False)
    inputs_fast = tf.reduce_mean(inputs_fast, axis=[1, 2, 3], keepdims=False)
    inputs = tf.concat([inputs_slow, inputs_fast], axis=1)
    inputs = self.dense(inputs)
    return tf.expand_dims(inputs, axis=1)

  def get_label_fn(self, batch_size=None):
    """Returns the function needed for adjusting label dims"""
    @tf.function
    def labels_with_batch_dim(labels):
      """Return last batch element"""
      return tf.slice(labels, [0, self.input_length-1], [batch_size, 1])
    @tf.function
    def labels_without_batch_dim(labels):
      """Return last element"""
      return tf.slice(labels, [self.input_length-1], [1])

    if batch_size is not None:
      return labels_with_batch_dim
    else:
      return labels_without_batch_dim

  def get_seq_length(self):
    return 1

  def get_seq_pool(self):
    return self.input_length

  def get_out_pool(self):
    return 1
