"""
CNN-LSTM architecture originally from
Kyritsis et al. (2019) https://ieeexplore.ieee.org/abstract/document/8606156
Adapted for end-to-end training without micromovement annotations by
Heydarian et al. (2020) https://ieeexplore.ieee.org/abstract/document/9187203
"""

import tensorflow as tf

class ConvLayer(tf.keras.layers.Layer):
  """One conv layer, potentially with max pooling"""

  def __init__(self, filters, kernel_size, use_pooling, l2_lambda):
    super(ConvLayer, self).__init__()
    self.use_pooling = use_pooling
    self.conv1d = tf.keras.layers.Conv1D(
      filters=filters, kernel_size=kernel_size, padding="same",
      kernel_initializer=tf.keras.initializers.VarianceScaling(),
      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    if self.use_pooling:
      self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2)

  @tf.function
  def call(self, inputs, training=False):
    inputs = self.conv1d(inputs)
    if self.use_pooling:
      inputs = self.max_pool(inputs)
    return inputs

class Model(tf.keras.Model):
  """Kyritsis CNN-LSTM Model"""

  def __init__(self, num_classes, input_length, specs, l2_lambda):
    super(Model, self).__init__()
    self.input_length = input_length
    self.seq_pool = specs["seq_pool"]
    self.conv_layers = []
    for i, (filters, kernel_size, use_pooling) in enumerate(specs["conv_layer_specs"]):
      self.conv_layers.append(ConvLayer(
        filters=filters, kernel_size=kernel_size, use_pooling=use_pooling,
        l2_lambda=l2_lambda))
    self.dense_1 = tf.keras.layers.Dense(units=5, activation="relu")
    self.lstm_1 = tf.keras.layers.LSTM(units=64, return_sequences=True,
      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    self.lstm_2 = tf.keras.layers.LSTM(units=64, return_sequences=True,
      kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    self.dense_2 = tf.keras.layers.Dense(units=num_classes, activation="sigmoid")

  @tf.function
  def call(self, inputs, training=False):
    for conv_layer in self.conv_layers:
      inputs = conv_layer(inputs, training=training)
    inputs = self.dense_1(inputs)
    inputs = self.lstm_1(inputs)
    inputs = self.lstm_2(inputs)
    inputs = self.dense_2(inputs)
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

  def get_out_pool(self):
    return self.seq_pool
