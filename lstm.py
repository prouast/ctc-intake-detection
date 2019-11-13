"""Simple LSTM Model"""

import tensorflow as tf

class Model(tf.keras.Model):

    """LSTM Model for fc7 data."""
    def __init__(self, num_classes, l2_lambda):
        super(Model, self).__init__()
        self.num_lstm = [64]
        self.lstm_blocks = []
        for i, num_units in enumerate(self.num_lstm):
            self.lstm_blocks.append(tf.keras.layers.LSTM(
                    units=num_units, return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)))
        self.dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def __call__(self, inputs, training=False):
        for lstm_block in self.lstm_blocks:
            inputs = lstm_block(inputs)
        inputs = self.dense(inputs)
        inputs = self.dropout(inputs)
        return inputs
