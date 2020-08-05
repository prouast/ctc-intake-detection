"""Pipeline for the Clemson dataset"""

import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from absl import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CHANNELS = 3
EVENT_NAMES_MAP = {
  "label_1": ["idle", "intake"],
  "label_2": ["idle", "bite", "drink"],
  "label_3": ["idle", "left", "right", "both"],
  "label_4": ["idle", "chopsticks", "fork", "hand", "knife", "spoon"],
  "label_5": ["idle", "bowl", "glass", "mug", "plate"],
  "label_6": ["idle", "dessert", "drink", "fruit_veg", "meat_dish", "pizza", "rice_dish", "salad", "sandwich_wrap", "snack", "soup_stew", "veg_dish"]}
NUM_TRAINING_FILES = 302
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]

logging.set_verbosity(logging.INFO)

class Dataset():

  def __init__(self, label_mode, input_length, seq_shift, def_val):
    self.label_mode = label_mode
    self.input_length = input_length
    self.seq_shift = seq_shift
    self.def_val = def_val

  def __get_hash_table(self, label_category):
    assert self.def_val >= 0, "def_val must be greater or equal 0"
    # Event names
    event_names = EVENT_NAMES_MAP[label_category]
    # Number of events including default event
    num_events = len(event_names)
    # Get the table
    table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
        keys=event_names,
        values=tf.range(self.def_val, num_events + self.def_val, 1)), -1)
    return table

  def __get_input_parser(self, table):
    """Return the input parser"""
    def input_parser(serialized_example):
      """The input parser"""
      features = tf.io.parse_single_example(
        serialized_example, {
          'example/{}'.format(self.label_mode): tf.io.FixedLenFeature([], dtype=tf.string),
          'example/acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
          'example/gyro': tf.io.FixedLenFeature([3], dtype=tf.float32)
      })
      label = tf.cast(table.lookup(features['example/{}'.format(self.label_mode)]), tf.int32)
      features = tf.stack(
        [features['example/acc'], features['example/gyro']], 0)
      features = tf.squeeze(tf.reshape(features, [-1, 6]))
      return features, label
    return input_parser

  def __get_sequence_batch_fn(self, is_predicting, is_training):
    """Return sliding batched dataset or batched dataset."""
    if is_training or is_predicting:
      shift = self.seq_shift
    else:
      shift = self.input_length
    return lambda dataset: dataset.window(
      size=self.input_length, shift=shift, drop_remainder=True).flat_map(
        lambda f_w, l_w: tf.data.Dataset.zip(
          (f_w.batch(self.input_length), l_w.batch(self.input_length))))

  def __get_feature_transformation_parser(self, is_training):
    """Return the data transformation parser."""
    def transformation_parser(inert_data, label_data):
      """Apply distortions to inertial sequences."""
      if is_training:
        # Random horizontal flip
        def _flip_inertial(inert_data):
          """Flip hands"""
          mult = tf.concat([FLIP_ACC, FLIP_GYRO], axis=0)
          inert_data = tf.math.multiply(inert_data, mult)
          return inert_data
        condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
        inert_data = tf.cond(pred=condition,
          true_fn=lambda: _flip_inertial(inert_data),
          false_fn=lambda: inert_data)
      return inert_data, label_data
    return transformation_parser

  def __get_batch_parser(self, is_predicting, label_fn, collapse_fn):
    """Get the parser for processing one batch"""
    def batch_parser_train(features, labels):
      """Prepare batch for tf custom loop"""
      # Transform labels via label_fn
      labels = label_fn(labels)
      # Collapse labels
      labels_c, labels_l, _ = collapse_fn(labels)
      return features, labels, labels_c, labels_l
    def batch_parser_predict(features, labels):
      """Prepare batch for prediction"""
      # Transform labels via label_fn
      labels = label_fn(labels)
      return features, labels
    if is_predicting:
      return batch_parser_predict
    else:
      return batch_parser_train

  def num_event_classes(self):
    """Return the number of classes, excluding the default event"""
    return len(EVENT_NAMES_MAP[self.label_mode]) - 1

  def __call__(self, batch_size, data_dir, is_predicting, is_training, label_fn, collapse_fn, num_shuffle=1000):
    """Return the dataset pipeline"""
    # Scan for training files
    if is_predicting:
      filenames = [data_dir]
    else:
      filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecord"))
    if not filenames:
      raise RuntimeError('No files found.')
    logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Lookup table for Labels
    table = self.__get_hash_table(self.label_mode)
    # Shuffle files if needed
    if is_training:
      files = files.shuffle(NUM_TRAINING_FILES)
    pipeline = lambda filename: (tf.data.TFRecordDataset(filename)
      .map(map_func=self.__get_input_parser(table),
        num_parallel_calls=AUTOTUNE)
      .apply(self.__get_sequence_batch_fn(is_predicting, is_training))
      .map(map_func=self.__get_feature_transformation_parser(is_training),
        num_parallel_calls=AUTOTUNE))
    dataset = files.interleave(pipeline, cycle_length=4)
    if is_training:
      dataset = dataset.shuffle(num_shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(map_func=self.__get_batch_parser(
      is_predicting, label_fn, collapse_fn), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
