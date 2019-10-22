import os
import math
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import rewriter_config_pb2
import best_checkpoint_exporter
from ctc import greedy_decode_with_indices
from ctc import evaluate_interval_detection
from metrics import pre_rec
from metrics import f1_metric
import video_small_cnn_lstm
import inert_small_cnn_lstm
import lstm

ORIGINAL_SIZE = 140
FRAME_SIZE = 128
NUM_CHANNELS = 3
NUM_SHARDS = 10
NUM_SHUFFLE = 5000
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    name='batch_size', default=16, help='Batch size used for training.')
tf.app.flags.DEFINE_float(
    name='decay_rate', default=0.92, help='Rate at which learning rate decays.')
tf.app.flags.DEFINE_string(
    name='eval_dir', default='data/raw/eval', help='Directory for eval data.')
tf.app.flags.DEFINE_enum(
    name='input_mode', default="video_raw", enum_values=["video_raw", "inert", "video_fc7"],
    help='What is the input mode')
tf.app.flags.DEFINE_integer(
    name='input_features', default=2048, help='Number of input features.')
tf.app.flags.DEFINE_integer(
    name='input_fps', default=8, help='Number of input frames per second.')
tf.app.flags.DEFINE_enum(
    name='mode', default="train_and_evaluate", enum_values=["train_and_evaluate", "predict"],
    help='What mode should tensorflow be started in')
tf.app.flags.DEFINE_string(
    name='model', default='video_small_cnn_lstm',
    help='Select the model: {lstm, video_small_cnn_lstm, inert_small_cnn_lstm}')
tf.app.flags.DEFINE_string(
    name='model_dir', default='run',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_integer(
    name='num_seq', default=396960, help='Number of training sequences.')
tf.app.flags.DEFINE_integer(
    name='seq_length', default=16,
    help='Number of sequence elements.')
tf.app.flags.DEFINE_integer(
    name='seq_pool', default=1, help='Factor of sequence pooling in the model.')
tf.app.flags.DEFINE_integer(
    name='seq_shift', default=1,
    help='Shift in sequence generation.')
tf.app.flags.DEFINE_string(
    name='train_dir', default='data/raw/train', help='Directory for training data.')
tf.app.flags.DEFINE_float(
    name='train_epochs', default=200, help='Number of training epochs.')

tf.logging.set_verbosity(tf.logging.INFO)


def run_experiment(arg=None):
    """Run the experiment."""

    steps_per_epoch = int(FLAGS.num_seq / FLAGS.batch_size * FLAGS.seq_shift / FLAGS.seq_length)
    max_steps = steps_per_epoch * FLAGS.train_epochs

    # Model parameters
    params = tf.contrib.training.HParams(
        adam_epsilon=1e-8,
        base_learning_rate=1e-4,
        batch_size=FLAGS.batch_size,
        data_format='channels_last',
        decay_rate=FLAGS.decay_rate,
        dropout=0.5,
        gradient_clipping_norm=10.0,
        num_classes=2,
        seq_length=FLAGS.seq_length,
        steps_per_epoch=steps_per_epoch)

    # Bugfix, see https://github.com/tensorflow/tensorflow/issues/23780
    session_config = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    session_config.graph_options.rewrite_options.arithmetic_optimization = off

    # Run config
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        session_config=session_config)

    # Define the estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=params,
        config=run_config)

    # Exporters
    best_exporter = best_checkpoint_exporter.BestCheckpointExporter(
        score_metric='metrics/f1',
        compare_fn=lambda x,y: x.score > y.score,
        sort_key_fn=lambda x: -x.score)

    # Training input_fn
    def train_input_fn():
        return input_fn(is_training=True, data_dir=FLAGS.train_dir)

    # Eval input_fn
    def eval_input_fn():
        return input_fn(is_training=False, data_dir=FLAGS.eval_dir)

    # Define the experiment
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        exporters=best_exporter,
        start_delay_secs=30,
        throttle_secs=20)

    # Start the experiment
    if FLAGS.mode == "train_and_evaluate":
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif FLAGS.mode == "predict":
        predict_and_export_csv(estimator, eval_input_fn, FLAGS.eval_dir)


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_predicting = mode == tf.estimator.ModeKeys.PREDICT

    # Model
    if FLAGS.model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(params)
    elif FLAGS.model == "inert_small_cnn_lstm":
        model = inert_small_cnn_lstm.Model(params)
    elif FLAGS.model == "lstm":
        model = lstm.Model(params)

    logits = model(features, is_training)

    # Update seq_length according to seq pooling
    if FLAGS.seq_pool > 1:
        seq_length = int(FLAGS.seq_length / FLAGS.seq_pool)
    else:
        seq_length = FLAGS.seq_length

    # Decode logits into predictions
    predictions, _ = greedy_decode_with_indices(logits, params.num_classes, seq_length)

    pred_export = {
        'classes': tf.reshape(predictions, [-1]),
        'logits': tf.reshape(logits, [-1, params.num_classes+1])}

    if is_predicting:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_export,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(pred_export)
            })

    # If seq pooling performed in model, slice the labels as well
    if FLAGS.seq_pool > 1:
        labels = tf.strided_slice(input_=labels,
            begin=[0, FLAGS.seq_pool-1],
            end=[FLAGS.batch_size, FLAGS.seq_length],
            strides=[1, FLAGS.seq_pool])
    labels = tf.reshape(labels, [params.batch_size, seq_length])

    def dense_to_sparse(input, eos_token=0):
        idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
        values = tf.gather_nd(input, idx)
        shape = tf.shape(input, out_type=tf.int64)
        sparse = tf.SparseTensor(idx, values, shape)
        return sparse

    # Calculate ctc loss from SparseTensor without collapsing labels
    seq_lengths = tf.fill([params.batch_size], seq_length)
    loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(labels, eos_token=-1),
        inputs=logits,
        sequence_length=seq_lengths,
        preprocess_collapse_repeated=True,
        ctc_merge_repeated=False,
        time_major=False)

    # Reduce loss to average
    loss = tf.reduce_mean(loss)

    if is_training:
        global_step = tf.train.get_or_create_global_step()

        def _decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=params.steps_per_epoch, decay_rate=params.decay_rate)

        # Learning rate
        learning_rate = _decay_fn(params.base_learning_rate, global_step)
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('training/learning_rate', learning_rate)

        # The optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grad_vars = optimizer.compute_gradients(loss)

        tf.summary.scalar("training/global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        # Clip gradients
        grads, vars = zip(*grad_vars)
        grads, _ = tf.clip_by_global_norm(grads, params.gradient_clipping_norm)
        grad_vars = list(zip(grads, vars))

        for grad, var in grad_vars:
            var_name = var.name.replace(":", "_")
            tf.summary.histogram("gradients/%s" % var_name, grad)
            tf.summary.scalar("gradient_norm/%s" % var_name, tf.global_norm([grad]))
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("training/clipped_global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

    else:
        train_op = None

    # Calculate metrics
    pre, rec, pre_op, rec_op = pre_rec(labels, predictions, seq_length, evaluate_interval_detection)
    f1, f1_op = f1_metric(labels, predictions, seq_length, evaluate_interval_detection)

    # Save metrics
    tf.summary.scalar('metrics/precision', pre_op)
    tf.summary.scalar('metrics/recall', rec_op)
    tf.summary.scalar('metrics/f1', f1_op)
    metrics = {
        'metrics/precision': (pre, pre_op),
        'metrics/recall': (rec, rec_op),
        'metrics/f1': (f1, f1_op)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def input_fn(is_training, data_dir):
    """Input pipeline"""
    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecords"))
    if not filenames:
        raise RuntimeError('No files found.')
    tf.logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Lookup table for Labels
    table = None
    if FLAGS.input_mode == 'inert':
        mapping_strings = tf.constant(["Idle", "Intake"])
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=mapping_strings)
        # Initialize table
        with tf.Session() as sess:
            sess.run(table.init)
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
        dataset = dataset.shuffle(NUM_SHUFFLE).repeat()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    return dataset


def _get_input_parser(table):

    def input_parser_video_fc7(serialized_example):
        features = tf.parse_single_example(
            serialized_example, {
                'example/label': tf.FixedLenFeature([], dtype=tf.int64),
                'example/fc7': tf.FixedLenFeature([FLAGS.input_features], dtype=tf.float32)
        })
        label = tf.cast(features['example/label'], tf.int32)
        fc7 = features['example/fc7']
        return fc7, label

    def input_parser_video_raw(serialized_example):
        features = tf.parse_single_example(
            serialized_example, {
                'example/label_1': tf.FixedLenFeature([], dtype=tf.int64),
                'example/image': tf.FixedLenFeature([], dtype=tf.string)
        })
        label = tf.cast(features['example/label_1'], tf.int32)
        image_data = tf.decode_raw(features['example/image'], tf.uint8)
        image_data = tf.cast(image_data, tf.float32)
        image_data = tf.reshape(image_data,
            [ORIGINAL_SIZE, ORIGINAL_SIZE, NUM_CHANNELS])
        return image_data, label

    def input_parser_inert(serialized_example):
        features = tf.parse_single_example(
            serialized_example, {
                'example/label_1': tf.FixedLenFeature([], dtype=tf.string),
                'example/dom_acc': tf.FixedLenFeature([3], dtype=tf.float32),
                'example/dom_gyro': tf.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_acc': tf.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_gyro': tf.FixedLenFeature([3], dtype=tf.float32)
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
    shift = FLAGS.seq_shift if is_training else FLAGS.seq_length
    if tf.__version__ < "1.13.1":
        return tf.contrib.data.sliding_window_batch(
            window_size=FLAGS.seq_length, window_shift=FLAGS.seq_shift)
    else:
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


def predict_and_export_csv(estimator, eval_input_fn, eval_dir):
    tf.logging.info("Working on {0}.".format(eval_dir))
    tf.logging.info("Starting prediction...")
    predictions = estimator.predict(input_fn=eval_input_fn)
    pred_list = list(itertools.islice(predictions, None))
    pred_index = list(map(lambda item: item["classes"], pred_list))
    pred_logits_0 = list(map(lambda item: item["logits"][0], pred_list))
    pred_logits_1 = list(map(lambda item: item["logits"][1], pred_list))
    pred_logits_2 = list(map(lambda item: item["logits"][2], pred_list))
    # Get labels and ids
    filenames = gfile.Glob(os.path.join(eval_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(filenames))
    elem = dataset.map(input_parser).make_one_shot_iterator().get_next()
    labels = []; sess = tf.Session()
    num = len(pred_list)
    for i in range(0, num):
        val = sess.run(elem)
        labels.append(val[1])
    name = os.path.normpath(eval_dir).split(os.sep)[-1]
    tf.logging.info("Writing {0} examples to {1}.csv...".format(num, name))
    pred_array = np.column_stack((labels, pred_index, pred_logits_0, pred_logits_1, pred_logits_2))
    np.savetxt("Ayy_{0}.csv".format(name), pred_array, delimiter=",", fmt=['%i','%i','%f','%f','%f'])


# Run
if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
