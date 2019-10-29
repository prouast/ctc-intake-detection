import math
import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow import keras
from ctc import greedy_decode_with_indices
import metrics
import video_small_cnn_lstm
import inert_small_cnn_lstm
import lstm

FRAME_SIZE = 128
GRADIENT_CLIPPING_NORM = 10.0
LR_BOUNDARIES = [6000, 18000, 30000]
LR_VALUES = [1e-3, 1e-4, 1e-5, 1e-6]
NUM_CHANNELS = 3
NUM_CLASSES = 2
NUM_SHARDS = 20
NUM_SHUFFLE = 10000
ORIGINAL_SIZE = 140

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    name='batch_size', default=16, help='Batch size used for training.')
flags.DEFINE_string(
    name='eval_dir', default='data/raw/eval', help='Directory for val data.')
flags.DEFINE_enum(
    name='input_mode', default="video_raw", enum_values=["video_raw", "inert", "video_fc7"],
    help='What is the input mode')
flags.DEFINE_integer(
    name='input_features', default=2048, help='Number of input features.')
flags.DEFINE_integer(
    name='input_fps', default=8, help='Number of input frames per second.')
flags.DEFINE_float(
    name='l2_lambda', default=1e-3, help='l2 regularization lambda.')
flags.DEFINE_float(
    name='lr_base', default=1e-3, help='Base learning rate.')
flags.DEFINE_enum(
    name='lr_decay_fn', default="exponential", enum_values=["exponential", "piecewise_constant"],
    help='What is the input mode')
flags.DEFINE_float(
    name='lr_decay_rate', default=0.92, help='Rate at which learning rate decays.')
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
    if FLAGS.model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(FLAGS.seq_length, NUM_CLASSES, FLAGS.l2_lambda)
    elif FLAGS.model == "inert_small_cnn_lstm":
        model = inert_small_cnn_lstm.Model(NUM_CLASSES, FLAGS.l2_lambda)
    elif FLAGS.model == "lstm":
        model = lstm.Model(NUM_CLASSES, FLAGS.l2_lambda)

    # Instantiate the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_base)

    # Get the datasets
    train_dataset = dataset(is_training=True, data_dir=FLAGS.train_dir)
    eval_dataset = dataset(is_training=False, data_dir=FLAGS.eval_dir)

    # Instantiate the metrics
    train_pre_metric = metrics.Precision(
        def_val=0, seq_length=int(FLAGS.seq_length / FLAGS.seq_pool))
    train_rec_metric = metrics.Recall(
        def_val=0, seq_length=int(FLAGS.seq_length / FLAGS.seq_pool))
    train_f1_metric = metrics.F1(
        def_val=0, seq_length=int(FLAGS.seq_length / FLAGS.seq_pool))

    # Set up log writer
    train_writer = tf.summary.create_file_writer("log/train")
    #eval_writer = tf.summary.create_file_writer("log/eval")

    # Iterate over epochs
    for epoch in range(FLAGS.train_epochs):
        logging.info('Starting epoch %d' % (epoch,))

        # Iterate over batches
        for step, (features, labels) in enumerate(train_dataset):

            # If seq_pool performed, adjust seq_length and labels
            if FLAGS.seq_pool > 1:
                seq_length = int(FLAGS.seq_length / FLAGS.seq_pool)
                labels = tf.strided_slice(input_=labels,
                    begin=[0, FLAGS.seq_pool-1],
                    end=[FLAGS.batch_size, FLAGS.seq_length],
                    strides=[1, FLAGS.seq_pool])
            else:
                seq_length = FLAGS.seq_length
            labels = tf.reshape(labels, [FLAGS.batch_size, seq_length])

            # Open a GradientTape to record the operations run during forward pass
            with tf.GradientTape() as tape:

                # Run the forward pass
                logits = model(features)

                def dense_to_sparse(input, eos_token=0):
                    idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
                    values = tf.gather_nd(input, idx)
                    shape = tf.shape(input, out_type=tf.int64)
                    sparse = tf.SparseTensor(idx, values, shape)
                    return sparse

                # Calculate ctc loss from SparseTensor without collapsing labels
                seq_lengths = tf.fill([FLAGS.batch_size], seq_length)
                loss = tf.compat.v1.nn.ctc_loss(
                    labels=dense_to_sparse(labels, eos_token=-1),
                    inputs=logits,
                    sequence_length=seq_lengths,
                    preprocess_collapse_repeated=True,
                    ctc_merge_repeated=False,
                    time_major=False)

                # Reduce loss to scalar
                loss = tf.reduce_mean(loss)
                loss += sum(model.losses)

            # Retrieve gradient with gradient tape
            grads = tape.gradient(loss, model.trainable_weights)

            # Apply the gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Decode logits into predictions
            predictions, _ = greedy_decode_with_indices(logits, NUM_CLASSES, seq_length)

            # Calculate metric
            train_pre_metric(labels, predictions)
            train_rec_metric(labels, predictions)
            train_f1_metric(labels, predictions)

            train_pre = train_pre_metric.result()
            train_rec = train_rec_metric.result()
            train_f1 = train_f1_metric.result()

            # Log every 200 batches.
            if step % 50 == 0:
                logging.info('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                logging.info('Seen so far: %s samples' % ((step + 1) * FLAGS.batch_size))
                logging.info('Training precision: %s' % (float(train_pre),))
                logging.info('Training recall: %s' % (float(train_rec),))
                logging.info('Training f1: %s' % (float(train_f1),))
                with train_writer.as_default():
                    tf.summary.scalar('metrics/precision', data=train_pre, step=step)
                    tf.summary.scalar('metrics/recall', data=train_rec, step=step)
                    tf.summary.scalar('metrics/f1', data=train_f1, step=step)
                    tf.summary.scalar('training/loss', data=loss, step=step)
                    train_writer.flush()

        # Display metrics at the end of each epoch.
        train_pre = train_pre_metric.result()
        logging.info('Training precision over epoch: %s' % (float(train_pre),))
        train_rec = train_rec_metric.result()
        logging.info('Training recall over epoch: %s' % (float(train_rec),))
        train_f1 = train_f1_metric.result()
        logging.info('Training f1 over epoch: %s' % (float(train_f1),))

        # Reset training metrics at the end of each epoch
        train_pre_metric.reset_states()
        train_rec_metric.reset_states()
        train_f1_metric.reset_states()

        # Evaluation
        #with eval_writer.as_default():
            #tf.summary.scalar('metrics/precision', data=train_pre, step=step)
            #tf.summary.scalar('metrics/recall', data=train_rec, step=step)
            #tf.summary.scalar('metrics/f1', data=train_f1, step=step)
            #tf.summary.scalar('training/loss', data=loss, step=step)

        writer.flush()

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
        files = files.shuffle(NUM_SHARDS)
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=_get_input_parser(table), num_parallel_calls=2)
            .apply(_get_sequence_batch_fn(is_training))
            .map(map_func=_get_transformation_parser(is_training),
                num_parallel_calls=2),
        cycle_length=NUM_SHARDS)
    if is_training:
        dataset = dataset.shuffle(NUM_SHUFFLE)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

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
    shift = 1 if is_training else FLAGS.seq_length
    return lambda dataset: dataset.window(
        size=FLAGS.seq_length, shift=shift, drop_remainder=True).flat_map(
            lambda f_w, l_w: tf.data.Dataset.zip(
                (f_w.batch(FLAGS.seq_length), l_w.batch(FLAGS.seq_length))))

def _get_transformation_parser(is_training):
    """Return the data transformation parser."""

    def transformation_parser(image_data, label_data):
        """Apply distortions to sequences."""

        if is_training:

            # Random rotation
            rotation_degree = tf.random_uniform([], -2.0, 2.0)
            rotation_radian = rotation_degree * math.pi / 180
            image_data = tf.contrib.image.rotate(image_data,
                angles=rotation_radian)

            # Random crop
            diff = ORIGINAL_SIZE - FRAME_SIZE + 1
            limit = [1, diff, diff, 1]
            offset = tf.random_uniform(shape=tf.shape(limit),
                dtype=tf.int32, maxval=tf.int32.max) % limit
            size = [FLAGS.seq_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
            image_data = tf.slice(image_data, offset, size)

            # Random horizontal flip
            condition = tf.less(tf.random_uniform([], 0, 1.0), .5)
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
            delta = tf.random_uniform([], -63, 63)
            image_data = _adjust_brightness(image_data, delta)

            # Random contrast change -
            def _adjust_contrast(image_data, contrast_factor):
                if tf.shape(image_data)[0] == 4:
                    contrast = lambda x: tf.image.adjust_contrast(x, contrast_factor)
                    return tf.map_fn(contrast, image_data)
                else:
                    return tf.image.adjust_contrast(image_data, contrast_factor)
            contrast_factor = tf.random_uniform([], 0.2, 1.8)
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

    if FLAGS.input_mode == "video_fc7":
        return lambda f, l: (f, l)
    elif FLAGS.input_mode == "video_raw":
        return transformation_parser
    elif FLAGS.input_mode == "inert":
        return lambda f, l: (f, l)

# Run
if __name__ == "__main__":
    app.run(main=run_experiment)
