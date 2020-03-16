"""Pipeline for the OREBA dataset"""

# Mode (video / inertial)
# label_mode
# input fps
# inpiut length
# seq fps

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
NUM_EVENT_CLASSES_MAP = {"label_1": 1, "label_2": 2, "label_3": 3, "label_4": 6}
FLIP_ACC = [1., -1., 1.]
FLIP_GYRO = [-1., 1., -1.]


class OREBA():

    def __init__(self, label_mode, input_mode, input_length, input_fps, seq_fps):
        self.label_mode = label_mode
        self.input_mode = input_mode
        self.input_length = input_length
        self.input_fps = input_fps
        self.seq_fps = seq_fps

    def __get_hash_table(self, label_category):
        if label_category == 'label_1':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "Intake"], [0, 1]), -1)
        elif label_category == 'label_2':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "Intake-Eat", "Intake-Drink"], [0, 1, 2]), -1)
        elif label_category == 'label_3':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "Right", "Left", "Both"], [0, 1, 2, 3]), -1)
        elif label_category == 'label_4':
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    ["Idle", "Hand", "Fork", "Spoon", "Knife", "Finger", "Cup"], [0, 1, 2, 3, 4, 5, 6]), -1)
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
            shift = int(self.input_fps/self.seq_fps)
        else:
            shift = self.input_length
        return lambda dataset: dataset.window(
            size=self.input_length, shift=shift, drop_remainder=True).flat_map(
                lambda f_w, l_w: tf.data.Dataset.zip(
                    (f_w.batch(self.input_length), l_w.batch(self.input_length))))

    def __get_transformation_parser(self, is_training):
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
                size = [FLAGS.input_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
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
                def _flip_inertial(inert_data):
                    """Flip hands"""
                    mult = tf.tile(tf.concat([FLIP_ACC, FLIP_GYRO], axis=0), [2])
                    inert_data = tf.math.multiply(inert_data, mult)
                    return tf.concat([inert_data[:, 6:12], inert_data[:, 0:6]], axis=1)
                condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
                inert_data = tf.cond(pred=condition,
                    true_fn=lambda: _flip_inertial(inert_data),
                    false_fn=lambda: inert_data)

            return inert_data, label_data

        if self.input_mode == "video":
            return image_transformation_parser
        elif self.input_mode == "inert":
            return inert_transformation_parser

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
            .map(map_func=self.__get_transformation_parser(is_training),
                num_parallel_calls=AUTOTUNE))
        dataset = files.interleave(pipeline, cycle_length=4)
        if is_training:
            dataset = dataset.shuffle(num_shuffle)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset
