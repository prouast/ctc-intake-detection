"""Pipeline for the Clemson dataset"""

import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from absl import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CHANNELS = 3
NUM_EVENT_CLASSES_MAP = {"label_1": 1, "label_2": 3, "label_3": 5, "label_4": 4}
NUM_TRAINING_FILES = 302
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]

logging.set_verbosity(logging.INFO)

class Dataset():

    def __init__(self, label_mode, input_length, input_fps, seq_fps):
        self.label_mode = label_mode
        self.input_length = input_length
        self.input_fps = input_fps
        self.seq_fps = seq_fps

    def __get_hash_table(self, label_category):
        if label_category == 'label_1':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "bite"], [0, 1]), -1)
        elif label_category == 'label_2':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "left", "right", "both"], [0, 1, 2, 3]), -1)
        elif label_category == 'label_3':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "chopsticks", "fork", "hand", "knife", "spoon"], [0, 1, 2, 3, 4, 5]), -1)
        elif label_category == 'label_4':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "bowl", "glass", "mug", "plate"], [0, 1, 2, 3, 4]), -1)
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

    def __get_sequence_batch_fn(self, is_training, is_predicting):
        """Return sliding batched dataset or batched dataset."""
        if is_training or is_predicting:
            shift = int(self.input_fps/self.seq_fps)
        else:
            shift = self.input_length
        return lambda dataset: dataset.window(
            size=self.input_length, shift=shift, drop_remainder=True).flat_map(
                lambda f_w, l_w: tf.data.Dataset.zip(
                    (f_w.batch(self.input_length), l_w.batch(self.input_length))))

    def __get_transformation_parser(self, is_training):
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

    def num_classes(self):
        return NUM_EVENT_CLASSES_MAP[self.label_mode]

    def __call__(self, batch_size, is_training, is_predicting, data_dir, num_shuffle=1000):
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
            .apply(self.__get_sequence_batch_fn(is_training, is_predicting))
            .map(map_func=self.__get_transformation_parser(is_training),
                num_parallel_calls=AUTOTUNE))
        dataset = files.interleave(pipeline, cycle_length=4)
        if is_training:
            dataset = dataset.shuffle(num_shuffle)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset
