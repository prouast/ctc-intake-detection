"""CNN-LSTM Model for OREBA-DIS dataset from Heydarian et al. (2020)"""

import tensorflow as tf

class Model(tf.keras.Model):
    """CNN-LSTM for inertial data"""

    def __init__(self, num_classes, seq_pool, l2_lambda):
        super(Model, self).__init__()
        # Make sure model implied seq_pool equals arg implied seq_pool
        assert seq_pool == 16, \
            "seq_pool: Model implied == 16 != {} == arg implied".format(seq_pool)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=1, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_3 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=5, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.conv1d_4 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=7, padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.lstm_1 = tf.keras.layers.LSTM(
            units=64, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.lstm_2 = tf.keras.layers.LSTM(
            units=64, return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dense_1 = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))

    @tf.function
    def __call__(self, inputs, training=False):
        inputs = self.conv1d_1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.max_pool(inputs)
        inputs = self.conv1d_2(inputs)
        inputs = self.dropout(inputs)
        inputs = self.max_pool(inputs)
        inputs = self.conv1d_3(inputs)
        inputs = self.dropout(inputs)
        inputs = self.max_pool(inputs)
        inputs = self.conv1d_4(inputs)
        inputs = self.dropout(inputs)
        inputs = self.max_pool(inputs)
        inputs = self.lstm_1(inputs)
        inputs = self.lstm_2(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dropout(inputs)
        return inputs
