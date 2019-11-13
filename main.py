import math
import os
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow import keras
from ctc import ctc_decode_logits
from ctc import ctc_loss
from model_saver import ModelSaver
import metrics
import video_small_cnn_lstm
import inert_small_cnn_lstm
import lstm

FRAME_SIZE = 128
LR_BOUNDARIES = [2, 7, 10]
LR_VALUE_DIV = [1., 10., 100., 1000.]
LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 1
FLIP_ACC = [1., -1., 1.]
FLIP_GYRO = [-1., 1., -1.]
NUM_CHANNELS = 3
NUM_CLASSES = 2
NUM_SHUFFLE = 100000
NUM_TRAINING_FILES = 62
ORIGINAL_SIZE = 140
AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    name='batch_size', default=16, help='Batch size used for training.')
flags.DEFINE_string(
    name='eval_dir', default='data/raw/eval', help='Directory for val data.')
flags.DEFINE_integer(
    name='eval_steps', default=250, help='Eval and save best model after every x steps.')
flags.DEFINE_enum(
    name='ctc_mode', default="naive", enum_values=["naive", "all_collapse", "selective_collapse"],
    help='What is the input mode')
flags.DEFINE_enum(
    name='input_mode', default="video_raw", enum_values=["video_raw", "inert", "video_fc7"],
    help='What is the input mode')
flags.DEFINE_integer(
    name='input_features', default=2048, help='Number of input features in fc7 mode.')
flags.DEFINE_integer(
    name='input_fps', default=8, help='Frames per seconds in input data.')
flags.DEFINE_float(
    name='l2_lambda', default=1e-3, help='l2 regularization lambda.')
flags.DEFINE_integer(
    name='log_steps', default=100, help='Log after every x steps.')
flags.DEFINE_float(
    name='lr_base', default=1e-3, help='Base learning rate.')
flags.DEFINE_enum(
    name='lr_decay_fn', default="exponential", enum_values=["exponential", "piecewise_constant"],
    help='What is the input mode')
flags.DEFINE_enum(
    name='mode', default="train_and_evaluate", enum_values=["train_and_evaluate", "predict"],
    help='What mode should tensorflow be started in')
flags.DEFINE_enum(
    name='model', default='video_small_cnn_lstm', enum_values=["lstm", "video_small_cnn_lstm", "inert_small_cnn_lstm"],
    help='Select the model: {lstm, video_small_cnn_lstm, inert_small_cnn_lstm}')
flags.DEFINE_string(
    name='model_dir', default='run',
    help='Output directory for model and training stats.')
flags.DEFINE_integer(
    name='seq_fps', default=8, help='Target frames per seconds in sequence generation.')
flags.DEFINE_integer(
    name='seq_length', default=16,
    help='Number of sequence elements.')
flags.DEFINE_integer(
    name='seq_pool', default=1, help='Factor of sequence pooling in the model.')
flags.DEFINE_string(
    name='train_dir', default='data/raw/train', help='Directory for training data.')
flags.DEFINE_integer(
    name='train_epochs', default=200, help='Number of training epochs.')

logging.set_verbosity(logging.INFO)

def run_experiment(arg=None):
    """Run the experiment."""

    # Get the model
    num_classes = NUM_CLASSES if FLAGS.ctc_mode == 'naive' else NUM_CLASSES + 1
    if FLAGS.model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(FLAGS.seq_length, num_classes, FLAGS.l2_lambda)
    elif FLAGS.model == "inert_small_cnn_lstm":
        model = inert_small_cnn_lstm.Model(num_classes, FLAGS.l2_lambda)
    elif FLAGS.model == "lstm":
        model = lstm.Model(num_classes, FLAGS.l2_lambda)

    # Instantiate learning rate schedule and optimizer
    if FLAGS.lr_decay_fn == "exponential":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FLAGS.lr_base,
            decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE, staircase=True)
    elif FLAGS.lr_decay_fn == "piecewise_constant":
        values = np.divide(FLAGS.lr_base, LR_VALUE_DIV)
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=LR_BOUNDARIES, values=values.tolist())
    optimizer = Adam(learning_rate=lr_schedule)

    # Get the datasets
    train_dataset = dataset(is_training=True, data_dir=FLAGS.train_dir)
    eval_dataset = dataset(is_training=False, data_dir=FLAGS.eval_dir)

    # Instantiate the metrics
    seq_length = int(FLAGS.seq_length / FLAGS.seq_pool)
    train_pre_metric = metrics.Precision(def_val=0, seq_length=seq_length)
    train_rec_metric = metrics.Recall(def_val=0, seq_length=seq_length)
    train_f1_metric = metrics.F1(def_val=0, seq_length=seq_length)
    eval_tp_fp1_fp2_fn_metric = metrics.TP_FP1_FP2_FN(def_val=0, seq_length=seq_length)
    eval_pre_metric = metrics.Precision(def_val=0, seq_length=seq_length)
    eval_rec_metric = metrics.Recall(def_val=0, seq_length=seq_length)
    eval_f1_metric = metrics.F1(def_val=0, seq_length=seq_length)

    # Set up log writer
    train_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "log/train"))
    eval_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "log/eval"))

    # Save best checkpoints in terms of f1
    model_saver = ModelSaver(os.path.join(FLAGS.model_dir, "checkpoints"),
        compare_fn=lambda x,y: x.score > y.score, sort_reverse=True)

    # Keep track of total global step
    global_step = 0

    # Iterate over epochs
    for epoch in range(FLAGS.train_epochs):
        logging.info('Starting epoch %d' % (epoch,))

        # Iterate over training batches
        for step, (train_features, train_labels) in enumerate(train_dataset):

            # Adjust seq_length and labels
            train_labels = adjust_labels(train_labels, FLAGS.seq_pool,
                FLAGS.seq_length, FLAGS.batch_size)

            # Open a GradientTape to record the operations run during forward pass
            with tf.GradientTape() as tape:
                # Run the forward pass
                train_logits = model(train_features, training=True)
                # The loss function
                train_loss = ctc_loss(train_labels, train_logits,
                    ctc_mode=FLAGS.ctc_mode, batch_size=FLAGS.batch_size,
                    seq_length=seq_length, def_val=0, pad_val=-1)
                grads = tape.gradient(train_loss, model.trainable_weights)
            # Apply the gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Decode logits into predictions
            train_predictions, decoded = ctc_decode_logits(train_logits,
                ctc_mode=FLAGS.ctc_mode, num_classes=num_classes, seq_length=seq_length)

            # Update metrics
            train_pre_metric(train_labels, train_predictions)
            train_rec_metric(train_labels, train_predictions)
            train_f1_metric(train_labels, train_predictions)

            # Log every FLAGS.log_steps steps.
            if global_step % FLAGS.log_steps == 0:
                # Get metrics
                train_pre = train_pre_metric.result()
                train_rec = train_rec_metric.result()
                train_f1 = train_f1_metric.result()
                # Console
                logging.info('Step %s in epoch %s; global step %s' % (step, epoch, global_step))
                logging.info('Seen this epoch: %s samples' % ((step + 1) * FLAGS.batch_size))
                logging.info('Training loss (this step): %s' % float(train_loss))
                logging.info('Training precision (this step): %s' % (float(train_pre),))
                logging.info('Training recall (this step): %s' % (float(train_rec),))
                logging.info('Training f1 (this step): %s' % (float(train_f1),))
                # TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar('metrics/precision', data=train_pre, step=global_step)
                    tf.summary.scalar('metrics/recall', data=train_rec, step=global_step)
                    tf.summary.scalar('metrics/f1', data=train_f1, step=global_step)
                    tf.summary.scalar('training/loss', data=train_loss, step=global_step)
                    tf.summary.scalar('training/learning_rate', data=lr_schedule(epoch), step=global_step)
                train_pre_metric.reset_states()
                train_rec_metric.reset_states()
                train_f1_metric.reset_states()
                train_writer.flush()

            # Evaluate every FLAGS.eval_steps steps.
            if global_step % FLAGS.eval_steps == 0:
                logging.info('Evaluating at global step %s' % global_step)

                # Keep track of eval losses
                eval_losses = []

                # Iterate through eval batches
                for i, (eval_features, eval_labels) in enumerate(eval_dataset):

                    # Adjust seq_length and labels
                    eval_labels = adjust_labels(eval_labels, FLAGS.seq_pool,
                        FLAGS.seq_length, FLAGS.batch_size)

                    # Run the forward pass
                    eval_logits = model(eval_features, training=False)
                    # The loss function
                    eval_loss = ctc_loss(eval_labels, eval_logits, ctc_mode=FLAGS.ctc_mode,
                            batch_size=FLAGS.batch_size, seq_length=seq_length,
                            def_val=0, pad_val=-1)
                    eval_losses.append(eval_loss.numpy())

                    # Decode logits into predictions
                    eval_predictions, decoded = ctc_decode_logits(eval_logits,
                        ctc_mode=FLAGS.ctc_mode, num_classes=num_classes,
                        seq_length=seq_length)

                    # Calculate metric
                    eval_tp_fp1_fp2_fn_metric(eval_labels, eval_predictions)
                    eval_pre_metric(eval_labels, eval_predictions)
                    eval_rec_metric(eval_labels, eval_predictions)
                    eval_f1_metric(eval_labels, eval_predictions)

                # Get metrics
                eval_tp, eval_fp1, eval_fp2, eval_fn = eval_tp_fp1_fp2_fn_metric.result()
                eval_pre = eval_pre_metric.result()
                eval_rec = eval_rec_metric.result()
                eval_f1 = eval_f1_metric.result()
                eval_loss = np.mean(eval_losses)

                # Console
                logging.info('Evaluation loss: %s' % float(eval_loss))
                logging.info('Evaluation tp: %s, fp1: %s, fp2: %s, fn: %s' %
                    (int(eval_tp), int(eval_fp1), int(eval_fp2), int(eval_fn)))
                logging.info('Evaluation precision: %s' % (float(eval_pre),))
                logging.info('Evaluation recall: %s' % (float(eval_rec),))
                logging.info('Evaluation f1: %s' % (float(eval_f1),))

                # TensorBoard
                with eval_writer.as_default():
                    tf.summary.scalar('metrics/tp', data=eval_tp, step=global_step)
                    tf.summary.scalar('metrics/fp_1', data=eval_fp1, step=global_step)
                    tf.summary.scalar('metrics/fp_2', data=eval_fp2, step=global_step)
                    tf.summary.scalar('metrics/fn', data=eval_fn, step=global_step)
                    tf.summary.scalar('metrics/precision', data=eval_pre, step=global_step)
                    tf.summary.scalar('metrics/recall', data=eval_rec, step=global_step)
                    tf.summary.scalar('metrics/f1', data=eval_f1, step=global_step)
                    tf.summary.scalar('training/loss', data=eval_loss, step=global_step)
                eval_writer.flush()

                # Reset eval metric states after evaluation
                eval_tp_fp1_fp2_fn_metric.reset_states()
                eval_pre_metric.reset_states()
                eval_rec_metric.reset_states()
                eval_f1_metric.reset_states()

                # Save best models
                model_saver.save(model=model, score=eval_f1.numpy(),
                    step=global_step, file="model")

            # Increment global step
            global_step += 1

        logging.info('Finished epoch %s' % (epoch,))
        optimizer.finish_epoch()

class Adam(keras.optimizers.Adam):
    """Adam optimizer that retrieves learning rate based on epochs"""
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self._epochs = None
    def _decayed_lr(self, var_dtype):
        """Get learning rate based on epochs."""
        lr_t = self._get_hyper("learning_rate", var_dtype)
        if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            epochs = tf.cast(self.epochs, var_dtype)
            lr_t = tf.cast(lr_t(epochs), var_dtype)
        return lr_t
    @property
    def epochs(self):
        """Variable. The number of epochs."""
        if self._epochs is None:
            self._epochs = self.add_weight(
                "epochs", shape=[], dtype=tf.int64, trainable=False,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
            self._weights.append(self._epochs)
        return self._epochs
    def finish_epoch(self):
        """Increment epoch count"""
        return self._epochs.assign_add(1)

def adjust_labels(labels, seq_pool, seq_length, batch_size):
    """If seq_pool performed, adjust seq_length and labels by slicing"""
    if seq_pool > 1:
        labels = tf.strided_slice(
            input_=labels, begin=[0, seq_pool-1], end=[batch_size, seq_length],
            strides=[1, seq_pool])
    labels = tf.reshape(labels, [batch_size, int(seq_length/seq_pool)])
    return labels

def dataset(is_training, data_dir):
    """Input pipeline"""
    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecords"))
    if not filenames:
        raise RuntimeError('No files found.')
    logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Lookup table for Labels
    table = None
    if FLAGS.input_mode == 'inert':
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                ["Idle", "Intake"], [0, 1]), -1)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_TRAINING_FILES)
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=_get_input_parser(table),
                num_parallel_calls=AUTOTUNE)
            .apply(_get_sequence_batch_fn(is_training))
            .map(map_func=_get_transformation_parser(is_training),
                num_parallel_calls=AUTOTUNE),
        cycle_length=4)
    if is_training:
        dataset = dataset.shuffle(NUM_SHUFFLE)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

def _get_input_parser(table):
    """Return the input parser"""

    def input_parser_video_fc7(serialized_example):
        """Parser for fc7 video features"""
        features = tf.io.parse_single_example(
            serialized_example, {
                'example/label': tf.io.FixedLenFeature([], dtype=tf.int64),
                'example/fc7': tf.io.FixedLenFeature([FLAGS.input_features], dtype=tf.float32)
        })
        label = tf.cast(features['example/label'], tf.int32)
        fc7 = features['example/fc7']
        return fc7, label

    def input_parser_video_raw(serialized_example):
        """Parser for raw video"""
        features = tf.io.parse_single_example(
            serialized_example, {
                'example/label_1': tf.io.FixedLenFeature([], dtype=tf.int64),
                'example/image': tf.io.FixedLenFeature([], dtype=tf.string)
        })
        label = tf.cast(features['example/label_1'], tf.int32)
        image_data = tf.decode_raw(features['example/image'], tf.uint8)
        image_data = tf.cast(image_data, tf.float32)
        image_data = tf.reshape(image_data,
            [ORIGINAL_SIZE, ORIGINAL_SIZE, NUM_CHANNELS])
        return image_data, label

    def input_parser_inert(serialized_example):
        """Parser for inertial data"""
        features = tf.io.parse_single_example(
            serialized_example, {
                'example/label_1': tf.io.FixedLenFeature([], dtype=tf.string),
                'example/dom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/dom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32)
        })
        label = tf.cast(table.lookup(features['example/label_1']), tf.int32)
        features = tf.stack(
            [features['example/dom_acc'], features['example/dom_gyro'],
             features['example/ndom_acc'], features['example/ndom_gyro']], 0)
        features = tf.squeeze(tf.reshape(features, [-1, 12]))
        return features, label

    if FLAGS.input_mode == "video_fc7":
        return input_parser_video_fc7
    elif FLAGS.input_mode == "video_raw":
        return input_parser_video_raw
    elif FLAGS.input_mode == "inert":
        return input_parser_inert

def _get_sequence_batch_fn(is_training):
    """Return sliding batched dataset or batched dataset."""
    shift = int(FLAGS.input_fps/FLAGS.seq_fps) if is_training else FLAGS.seq_length
    return lambda dataset: dataset.window(
        size=FLAGS.seq_length, shift=shift, drop_remainder=True).flat_map(
            lambda f_w, l_w: tf.data.Dataset.zip(
                (f_w.batch(FLAGS.seq_length), l_w.batch(FLAGS.seq_length))))

def _get_transformation_parser(is_training):
    """Return the data transformation parser."""

    def image_transformation_parser(image_data, label_data):
        """Apply distortions to image sequences."""

        if is_training:

            # Random rotation
            rotation_degree = tf.random.uniform([], -2.0, 2.0)
            rotation_radian = rotation_degree * math.pi / 180
            image_data = tf.contrib.image.rotate(image_data,
                angles=rotation_radian)

            # Random crop
            diff = ORIGINAL_SIZE - FRAME_SIZE + 1
            limit = [1, diff, diff, 1]
            offset = tf.random.uniform(shape=tf.shape(limit),
                dtype=tf.int32, maxval=tf.int32.max) % limit
            size = [FLAGS.seq_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
            image_data = tf.slice(image_data, offset, size)

            # Random horizontal flip
            condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
            image_data = tf.cond(pred=condition,
                true_fn=lambda: tf.image.flip_left_right(image_data),
                false_fn=lambda: image_data)

            # Random brightness change
            def _adjust_brightness(image_data, delta):
                if tf.shape(image_data)[0] == 4:
                    brightness = lambda x: tf.image.adjust_brightness(x, delta)
                    return tf.map_fn(brightness, image_data)
                else:
                    return tf.image.adjust_brightness(image_data, delta)
            delta = tf.random.uniform([], -63, 63)
            image_data = _adjust_brightness(image_data, delta)

            # Random contrast change -
            def _adjust_contrast(image_data, contrast_factor):
                if tf.shape(image_data)[0] == 4:
                    contrast = lambda x: tf.image.adjust_contrast(x, contrast_factor)
                    return tf.map_fn(contrast, image_data)
                else:
                    return tf.image.adjust_contrast(image_data, contrast_factor)
            contrast_factor = tf.random.uniform([], 0.2, 1.8)
            image_data = _adjust_contrast(image_data, contrast_factor)

        else:

            # Crop the central [height, width].
            image_data = tf.image.resize_image_with_crop_or_pad(
                image_data, FRAME_SIZE, FRAME_SIZE)
            size = [FLAGS.seq_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
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
            min_stddev = tf.rsqrt(tf.cast(num_pixels, dtype=tf.float32))
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

    if FLAGS.input_mode == "video_fc7":
        return lambda f, l: (f, l)
    elif FLAGS.input_mode == "video_raw":
        return image_transformation_parser
    elif FLAGS.input_mode == "inert":
        return inert_transformation_parser

# Run
if __name__ == "__main__":
    app.run(main=run_experiment)
