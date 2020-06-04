"""Pipeline for the OREBA dataset"""

import math
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.platform import gfile
from absl import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_TRAINING_FILES = 62
ORIGINAL_SIZE = 140
FRAME_SIZE = 128
NUM_CHANNELS = 3
EVENT_NAMES_MAP = {
  "label_1": ["Idle", "Intake"],
  "label_2": ["Idle", "Intake-Eat", "Intake-Drink"],
  "label_3": ["Idle", "Right", "Left", "Both"],
  "label_4": ["Idle", "Hand", "Fork", "Spoon", "Knife", "Finger", "Cup"]}
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]
FLIP_ORIENT = [-1., -1., 1.]

logging.set_verbosity(logging.INFO)

class Dataset():

  def __init__(self, label_mode, input_mode, input_length, seq_shift, def_val):
    self.label_mode = label_mode
    self.input_mode = input_mode
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
    def input_parser_video(serialized_example):
      """Parser for raw video"""
      features = tf.io.parse_single_example(
        serialized_example, {
          'example/{}'.format(self.label_mode):
            tf.io.FixedLenFeature([], dtype=tf.int64),
          'example/image':
            tf.io.FixedLenFeature([], dtype=tf.string)
      })
      label = tf.cast(features['example/{}'.format(self.label_mode)], tf.int32)
      image_data = tf.io.decode_raw(features['example/image'], tf.uint8)
      image_data = tf.cast(image_data, tf.float32)
      image_data = tf.reshape(image_data,
        [ORIGINAL_SIZE, ORIGINAL_SIZE, NUM_CHANNELS])
      image_data = tf.divide(image_data, 255) # Convert to [0, 1] range
      return image_data, label
    def input_parser_inert(serialized_example):
      """Parser for inertial data"""
      features = tf.io.parse_single_example(
        serialized_example, {
          'example/{}'.format(self.label_mode): tf.io.FixedLenFeature([], dtype=tf.string),
          'example/dom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
          'example/dom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32),
          'example/ndom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
          'example/ndom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32)
      })
      label = tf.cast(table.lookup(features['example/{}'.format(self.label_mode)]), tf.int32)
      features = tf.stack(
        [features['example/dom_acc'], features['example/dom_gyro'],
         features['example/ndom_acc'], features['example/ndom_gyro']], 0)
      features = tf.squeeze(tf.reshape(features, [-1, 12]))
      return features, label

    if self.input_mode == "video":
      return input_parser_video
    elif self.input_mode == "inert":
      return input_parser_inert

  def __get_sequence_batch_fn(self, is_training, is_predicting):
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

    def image_transformation_parser(image_data, label_data):
      """Apply distortions to image sequences."""

      if is_training:

        # Random rotation
        rotation_degree = tf.random.uniform([], -2.0, 2.0)
        rotation_radian = rotation_degree * math.pi / 180
        image_data = tfa.image.rotate(image_data,
          angles=rotation_radian)

        # Random crop
        diff = ORIGINAL_SIZE - FRAME_SIZE + 1
        limit = [1, diff, diff, 1]
        offset = tf.random.uniform(shape=tf.shape(limit),
          dtype=tf.int32, maxval=tf.int32.max) % limit
        size = [self.input_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
        image_data = tf.slice(image_data, offset, size)

        # Random horizontal flip
        condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
        image_data = tf.cond(pred=condition,
          true_fn=lambda: tf.image.flip_left_right(image_data),
          false_fn=lambda: image_data)

        # Random brightness change
        delta = tf.random.uniform([], -0.2, 0.2)
        image_data = tf.image.adjust_brightness(image_data, delta)

        # Random contrast change -
        contrast_factor = tf.random.uniform([], 0.8, 1.2)
        image_data = tf.image.adjust_contrast(image_data, 1.2)

      else:

        # Crop the central [height, width].
        image_data = tf.image.resize_with_crop_or_pad(
          image_data, FRAME_SIZE, FRAME_SIZE)
        size = [self.input_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
        image_data.set_shape(size)

      # Subtract off the mean and divide by the variance of the pixels.
      def _standardization(images):
        """Linearly scales image data to have zero mean and unit variance."""
        num_pixels = tf.reduce_prod(tf.shape(images))
        images_mean = tf.reduce_mean(images)
        variance = tf.reduce_mean(tf.square(images)) - tf.square(images_mean)
        variance = tf.nn.relu(variance)
        stddev = tf.sqrt(variance)
        # Apply a minimum normalization that protects us against uniform images.
        min_stddev = tf.math.rsqrt(tf.cast(num_pixels, dtype=tf.float32))
        pixel_value_scale = tf.maximum(stddev, min_stddev)
        pixel_value_offset = images_mean
        images = tf.subtract(images, pixel_value_offset)
        images = tf.divide(images, pixel_value_scale)
        return images
      image_data = _standardization(image_data)

      return image_data, label_data

    def inert_transformation_parser(inert_data, label_data):
      """Apply distortions to inertial sequences."""

      if is_training:
        # Random horizontal flip
        def _flip_hands(inert_data):
          """Flip hands"""
          # Derive multiplier
          flip = tf.concat([FLIP_ACC, FLIP_GYRO], axis=0)
          orient = tf.concat([FLIP_ORIENT, FLIP_ORIENT], axis=0)
          mult = tf.tile(tf.math.multiply(flip, orient), [2])
          # Transform values
          inert_data = tf.math.multiply(inert_data, mult)
          # Change indices
          inert_data = tf.concat([inert_data[:, 6:12], inert_data[:, 0:6]], axis=1)
          return inert_data
        condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
        inert_data = tf.cond(pred=condition,
          true_fn=lambda: _flip_hands(inert_data),
          false_fn=lambda: inert_data)
        # Random x-z rotation
        def _random_rotation(inert_data, rotation_0_1):
          """Simulate rotating sensor around wrist (y axis)"""
          # Derive multiplication matrix
          mult_0 = math.cos(rotation_0_1 * math.pi)
          mult_1 = math.sin(rotation_0_1 * math.pi)
          mult = tf.concat(
            [[mult_0, 0.0, -mult_1, 0.0,  0.0, 0.0],
             [0.0,  1.0, 0.0,  0.0,  0.0, 0.0],
             [mult_1, 0.0, mult_0, 0.0,  0.0, 0.0],
             [0.0,  0.0, 0.0,  mult_0, 0.0, -mult_1],
             [0.0,  0.0, 0.0,  0.0,  1.0, 0.0],
             [0.0,  0.0, 0.0,  mult_1, 0.0, mult_0]], axis=0)
          mult = tf.reshape(mult, [6, 6])
          # Rotation
          inert_data_left = tf.linalg.matmul(inert_data[:, 0:6], mult)
          inert_data_right = tf.linalg.matmul(inert_data[:, 6:12], mult)
          inert_data = tf.concat([inert_data_left, inert_data_right], axis=1)
          return inert_data
        # Do a random rotation between 0 and 180 degrees in 10% of cases
        rotation_degree = tf.cond(
          pred=tf.math.less(tf.random.uniform([], 0.0, 1.0), .1),
          true_fn=lambda: tf.random.uniform([], 0.0, 1.0),
          false_fn=lambda: tf.constant(0)])
        inert_data = _random_rotation(inert_data, rotation_degree)
        # Random orientation change
        def _change_orientation(inert_data, change_left, change_right):
          """Change orientation"""
          # Derive multiplier
          left_orient = FLIP_ORIENT if change_left else [1.0, 1.0, 1.0]
          right_orient = FLIP_ORIENT if change_right else [1.0, 1.0, 1.0]
          mult = tf.tile(tf.concat([left_orient, right_orient], axis=0), [2])
          # Transform values
          inert_data = tf.math.multiply(inert_data, mult)
          return inert_data
        # Disable for now
        #change_left = tf.less(tf.random.uniform([], 0, 1.0), .1)
        #inert_data = tf.cond(pred=condition,
        #  true_fn=lambda: _change_orientation(inert_data, True, False),
        #  false_fn=lambda: inert_data)
        #change_right = tf.less(tf.random.uniform([], 0, 1.0), .1)
        #inert_data = tf.cond(pred=condition,
        #  true_fn=lambda: _change_orientation(inert_data, False, True),
        #  false_fn=lambda: inert_data)

      return inert_data, label_data

    if self.input_mode == "video":
      return image_transformation_parser
    elif self.input_mode == "inert":
      return inert_transformation_parser

  def __get_batch_parser(self, label_fn, collapse_fn):
    """Get the parser for processing one batch"""
    def batch_parser(features, labels):
      # 1. Transform labels via label_fn
      labels = label_fn(labels)
      # 2. Collapse labels
      labels_c, labels_l = collapse_fn(labels)
      return features, labels, labels_c, labels_l
    return batch_parser

  def num_classes(self):
    """Return the number of classes, excluding the default event"""
    return len(EVENT_NAMES_MAP[self.label_mode]) - 1

  def __call__(self, batch_size, is_training, is_predicting, data_dir, label_fn, collapse_fn, num_shuffle=1000):
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
    table = None
    if self.input_mode == 'inert':
      table = self.__get_hash_table(self.label_mode)
    # Shuffle files if needed
    if is_training:
      files = files.shuffle(NUM_TRAINING_FILES)
    pipeline = lambda filename: (tf.data.TFRecordDataset(filename)
      .map(map_func=self.__get_input_parser(table),
        num_parallel_calls=AUTOTUNE)
      .apply(self.__get_sequence_batch_fn(is_training, is_predicting))
      .map(map_func=self.__get_feature_transformation_parser(is_training),
        num_parallel_calls=AUTOTUNE))
    if is_training:
      dataset = files.interleave(pipeline, cycle_length=4,
        num_parallel_calls=AUTOTUNE).shuffle(num_shuffle)
    else:
      dataset = files.flat_map(pipeline)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(map_func=self.__get_batch_parser(
      label_fn, collapse_fn), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
