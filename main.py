import math
import os
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.platform import gfile
from tensorflow import keras
from ctc import decode
from ctc import loss
from ctc import collapse
from model_saver import ModelSaver
import metrics
import video_small_cnn_lstm
import inert_small_cnn_lstm
import inert_heydarian_cnn_lstm

# Raw data
ORIGINAL_SIZE = 140
FRAME_SIZE = 128
NUM_CHANNELS = 3
NUM_EVENT_CLASSES_MAP = {"label_1": 1, "label_2": 2, "label_3": 3, "label_4": 6}
NUM_TRAINING_FILES = 62
FLIP_ACC = [1., -1., 1.]
FLIP_GYRO = [-1., 1., -1.]

# Representation
BLANK_INDEX = 0
DEF_VAL = 0
PAD_VAL = -1

# Hyperparameters
L2_LAMBDA = 1e-3
LR_BOUNDARIES = [5, 10, 15]
LR_VALUE_DIV = [1., 10., 100., 1000.]
LR_DECAY_RATE = 0.85
LR_DECAY_STEPS = 1
NUM_SHUFFLE = 50000
AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS
flags.DEFINE_integer(name='batch_size',
    default=128, help='Batch size used for training.')
flags.DEFINE_string(name='eval_dir',
    default='data/inert/eval', help='Directory for val data.')
flags.DEFINE_integer(name='eval_steps',
    default=250, help='Eval and save best model after every x steps.')
flags.DEFINE_integer(name='input_length',
    default=128, help='Number of input sequence elements.')
flags.DEFINE_enum(name='input_mode',
    default="inert", enum_values=["video", "inert"],
    help='What is the input mode')
flags.DEFINE_integer(name='input_fps',
    default=64, help='Frames per seconds in input data.')
flags.DEFINE_enum(name='label_mode',
    default="label_1", enum_values=NUM_EVENT_CLASSES_MAP.keys(),
    help='What is the label mode')
flags.DEFINE_integer(name='log_steps',
    default=100, help='Log after every x steps.')
flags.DEFINE_enum(name='loss_mode',
    default="ctc", enum_values=["ctc", "naive", "crossent"],
    help='What is the loss mode')
flags.DEFINE_float(name='lr_base',
    default=1e-3, help='Base learning rate.')
flags.DEFINE_enum(name='lr_decay_fn',
    default="exponential", enum_values=["exponential", "piecewise_constant"],
    help='What is the input mode')
flags.DEFINE_enum(name='mode',
    default="train_and_evaluate", enum_values=["train_and_evaluate", "predict"],
    help='What mode should tensorflow be started in')
flags.DEFINE_enum(name='model',
    default='inert_small_cnn_lstm', enum_values=["video_small_cnn_lstm", "inert_small_cnn_lstm", "inert_heydarian_cnn_lstm"],
    help='Select the model: {video_small_cnn_lstm, inert_small_cnn_lstm, inert_heydarian_cnn_lstm}')
flags.DEFINE_string(name='model_ckpt',
    default='run', help='Model checkpoint for prediction (e.g., model_5000).')
flags.DEFINE_string(name='model_dir',
    default='run', help='Output directory for model and training stats.')
flags.DEFINE_enum(name='predict_mode',
    default='batch_level_voted', enum_values=['video_level', 'batch_level', 'batch_level_voted'],
    help='How should the predictions be aggregated?')
flags.DEFINE_integer(name='seq_fps',
    default=8, help='Target frames per seconds in sequence generation.')
flags.DEFINE_string(name='train_dir',
    default='data/inert/train', help='Directory for training data.')
flags.DEFINE_integer(name='train_epochs',
    default=200, help='Number of training epochs.')

logging.set_verbosity(logging.INFO)

def train_and_evaluate():
    """Run the experiment."""

    # Read the representation choice
    seq_pool = int(FLAGS.input_fps/FLAGS.seq_fps)
    num_event_classes = _get_num_classes(FLAGS.label_mode)
    num_classes = num_event_classes + 1
    # Read the model choice
    if FLAGS.model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(num_classes, L2_LAMBDA)
    elif FLAGS.model == "inert_small_cnn_lstm":
        model = inert_small_cnn_lstm.Model(num_classes, L2_LAMBDA)
    elif FLAGS.model == "inert_heydarian_cnn_lstm":
        model = inert_heydarian_cnn_lstm.Model(num_classes, L2_LAMBDA)
    elif FLAGS.model == "lstm":
        model = lstm.Model(num_classes, L2_LAMBDA)

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

    # Get train and eval dataset
    train_dataset = dataset(is_training=True, is_predicting=False, data_dir=FLAGS.train_dir)
    eval_dataset = dataset(is_training=False, is_predicting=False, data_dir=FLAGS.eval_dir)

    # Instantiate the metrics
    seq_length = int(FLAGS.input_length / seq_pool)
    train_metrics = {
        'mean_precision': keras.metrics.Mean(),
        'mean_recall': keras.metrics.Mean(),
        'mean_f1': keras.metrics.Mean()}
    eval_metrics = {
        'mean_precision': keras.metrics.Mean(),
        'mean_recall': keras.metrics.Mean(),
        'mean_f1': keras.metrics.Mean()}
    for i in range(1, num_event_classes + 1):
        other_vals = [j for j in range(1, num_event_classes + 1) if j != i]
        train_metrics['class_{}_precision'.format(i)] = metrics.Precision(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        train_metrics['class_{}_recall'.format(i)] = metrics.Recall(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        train_metrics['class_{}_f1'.format(i)] = metrics.F1(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        eval_metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)] = metrics.TP_FP1_FP2_FP3_FN(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        eval_metrics['class_{}_precision'.format(i)] = metrics.Precision(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        eval_metrics['class_{}_recall'.format(i)] = metrics.Recall(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
        eval_metrics['class_{}_f1'.format(i)] = metrics.F1(
            event_val=i, def_val=DEF_VAL, seq_length=seq_length,
            other_vals=other_vals)
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

            # Adjust labels by slicing if necessary
            train_labels = _adjust_labels(train_labels, seq_pool,
                FLAGS.input_length, FLAGS.batch_size)

            # Open a GradientTape to record the operations run during forward pass
            with tf.GradientTape() as tape:
                # Run the forward pass
                train_logits = model(train_features, training=True)
                # The loss function
                train_loss = loss(train_labels, train_logits,
                    loss_mode=FLAGS.loss_mode, batch_size=FLAGS.batch_size,
                    seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL,
                    blank_index=BLANK_INDEX)
                grads = tape.gradient(train_loss, model.trainable_weights)
            # Apply the gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Decode logits into predictions
            train_predictions_u, train_predictions = decode(train_logits,
                loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                blank_index=BLANK_INDEX, def_val=DEF_VAL)
            train_predictions_u = collapse(train_predictions_u,
                seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL)

            # Log every FLAGS.log_steps steps.
            if global_step % FLAGS.log_steps == 0:
                # Update metrics
                for i in range(1, num_event_classes + 1):
                    train_metrics['class_{}_precision'.format(i)](train_labels, train_predictions_u)
                    train_metrics['class_{}_recall'.format(i)](train_labels, train_predictions_u)
                    train_metrics['class_{}_f1'.format(i)](train_labels, train_predictions_u)
                    train_metrics['mean_precision'](train_metrics['class_{}_precision'.format(i)].result())
                    train_metrics['mean_recall'](train_metrics['class_{}_recall'.format(i)].result())
                    train_metrics['mean_f1'](train_metrics['class_{}_f1'.format(i)].result())
                # General
                logging.info('Step %s in epoch %s; global step %s' % (step, epoch, global_step))
                logging.info('Seen this epoch: %s samples' % ((step + 1) * FLAGS.batch_size))
                logging.info('Training loss (this step): %s' % float(train_loss))
                logging.info('Mean training precision (this step): {}'.format(float(train_metrics['mean_precision'].result())))
                logging.info('Mean training recall (this step): {}'.format(float(train_metrics['mean_recall'].result())))
                logging.info('Mean training f1 (this step): {}'.format(float(train_metrics['mean_f1'].result())))
                # TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar('training/loss', data=train_loss, step=global_step)
                    tf.summary.scalar('training/learning_rate', data=lr_schedule(epoch), step=global_step)
                    tf.summary.scalar('metrics/mean_precision', data=train_metrics['mean_precision'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_recall', data=train_metrics['mean_recall'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_f1', data=train_metrics['mean_f1'].result(), step=global_step)
                # Reset metrics
                train_metrics['mean_precision'].reset_states()
                train_metrics['mean_recall'].reset_states()
                train_metrics['mean_f1'].reset_states()
                # For each class
                for i in range(1, num_event_classes + 1):
                    # Get metrics
                    train_pre = train_metrics['class_{}_precision'.format(i)].result()
                    train_rec = train_metrics['class_{}_recall'.format(i)].result()
                    train_f1 = train_metrics['class_{}_f1'.format(i)].result()
                    # Console
                    logging.info('Class {} training precision (this step): {}'.format(i, float(train_pre)))
                    logging.info('Class {} training recall (this step): {}'.format(i, float(train_rec)))
                    logging.info('Class {} training f1 (this step): {}'.format(i, float(train_f1)))
                    # TensorBoard
                    with train_writer.as_default():
                        tf.summary.scalar('metrics/class_{}_precision'.format(i), data=train_pre, step=global_step)
                        tf.summary.scalar('metrics/class_{}_recall'.format(i), data=train_rec, step=global_step)
                        tf.summary.scalar('metrics/class_{}_f1'.format(i), data=train_f1, step=global_step)
                    # Reset metrics
                    train_metrics['class_{}_precision'.format(i)].reset_states()
                    train_metrics['class_{}_recall'.format(i)].reset_states()
                    train_metrics['class_{}_f1'.format(i)].reset_states()
                # TensorBoard
                train_writer.flush()

            # Evaluate every FLAGS.eval_steps steps.
            if global_step % FLAGS.eval_steps == 0:
                logging.info('Evaluating at global step %s' % global_step)

                # Keep track of eval losses
                eval_losses = []

                # Iterate through eval batches
                for i, (eval_features, eval_labels) in enumerate(eval_dataset):

                    # Adjust seq_length and labels
                    eval_labels = _adjust_labels(eval_labels, seq_pool,
                        FLAGS.input_length, FLAGS.batch_size)

                    # Run the forward pass
                    eval_logits = model(eval_features, training=False)
                    # The loss function
                    eval_loss = loss(eval_labels, eval_logits,
                            loss_mode=FLAGS.loss_mode,
                            batch_size=FLAGS.batch_size, seq_length=seq_length,
                            def_val=DEF_VAL, pad_val=PAD_VAL,
                            blank_index=BLANK_INDEX)
                    eval_losses.append(eval_loss.numpy())

                    # Decode logits into predictions
                    eval_predictions_u, eval_predictions = decode(eval_logits,
                        loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                        blank_index=BLANK_INDEX, def_val=DEF_VAL)
                    eval_predictions_u = collapse(eval_predictions_u,
                        seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL)

                    # Update metric
                    for i in range(1, num_event_classes + 1):
                        eval_metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_precision'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_recall'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_f1'.format(i)](eval_labels, eval_predictions_u)

                for i in range(1, num_event_classes + 1):
                    eval_metrics['mean_precision'](eval_metrics['class_{}_precision'.format(i)].result())
                    eval_metrics['mean_recall'](eval_metrics['class_{}_recall'.format(i)].result())
                    eval_metrics['mean_f1'](eval_metrics['class_{}_f1'.format(i)].result())

                # Console
                eval_loss = np.mean(eval_losses)
                logging.info('Evaluation loss: %s' % float(eval_loss))
                logging.info('Mean eval precision: {}'.format(float(eval_metrics['mean_precision'].result())))
                logging.info('Mean eval recall: {}'.format(float(eval_metrics['mean_recall'].result())))
                logging.info('Mean eval f1: {}'.format(float(eval_metrics['mean_f1'].result())))
                # TensorBoard
                with eval_writer.as_default():
                    tf.summary.scalar('training/loss', data=eval_loss, step=global_step)
                    tf.summary.scalar('metrics/mean_precision', data=eval_metrics['mean_precision'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_recall', data=eval_metrics['mean_recall'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_f1', data=eval_metrics['mean_f1'].result(), step=global_step)
                # Save best models
                model_saver.save(model=model, score=float(eval_metrics['mean_f1'].result()), step=global_step, file="model")
                # Reset metrics
                eval_metrics['mean_precision'].reset_states()
                eval_metrics['mean_recall'].reset_states()
                eval_metrics['mean_f1'].reset_states()
                # For each class
                for i in range(1, num_event_classes + 1):
                    # Get metrics
                    eval_tp, eval_fp1, eval_fp2, eval_fp3, eval_fn = eval_metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)].result()
                    eval_pre = eval_metrics['class_{}_precision'.format(i)].result()
                    eval_rec = eval_metrics['class_{}_recall'.format(i)].result()
                    eval_f1 = eval_metrics['class_{}_f1'.format(i)].result()
                    # Console
                    logging.info('Class {} eval tp: {}, fp1: {}, fp2: {}, fp3: {}, fn: {}'.format(
                        i, int(eval_tp), int(eval_fp1), int(eval_fp2), int(eval_fp3), int(eval_fn)))
                    logging.info('Class {} eval precision: {}'.format(i, float(eval_pre)))
                    logging.info('Class {} eval recall: {}'.format(i, float(eval_rec)))
                    logging.info('Class {} eval f1: {}'.format(i, float(eval_f1)))
                    # TensorBoard
                    with eval_writer.as_default():
                        tf.summary.scalar('metrics/class_{}_tp'.format(i), data=eval_tp, step=global_step)
                        tf.summary.scalar('metrics/class_{}_fp_1'.format(i), data=eval_fp1, step=global_step)
                        tf.summary.scalar('metrics/class_{}_fp_2'.format(i), data=eval_fp2, step=global_step)
                        tf.summary.scalar('metrics/class_{}_fp_3'.format(i), data=eval_fp3, step=global_step)
                        tf.summary.scalar('metrics/class_{}_fn'.format(i), data=eval_fn, step=global_step)
                        tf.summary.scalar('metrics/class_{}_precision'.format(i), data=eval_pre, step=global_step)
                        tf.summary.scalar('metrics/class_{}_recall'.format(i), data=eval_rec, step=global_step)
                        tf.summary.scalar('metrics/class_{}_f1'.format(i), data=eval_f1, step=global_step)
                    # Reset eval metric states after evaluation
                    eval_metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)].reset_states()
                    eval_metrics['class_{}_precision'.format(i)].reset_states()
                    eval_metrics['class_{}_recall'.format(i)].reset_states()
                    eval_metrics['class_{}_f1'.format(i)].reset_states()
                # TensorBoard
                eval_writer.flush()

            # Increment global step
            global_step += 1

        logging.info('Finished epoch %s' % (epoch,))
        optimizer.finish_epoch()

def predict():
    assert FLAGS.batch_size == 1, "batch_size should be 1 for prediction"
    # Get the model
    seq_pool = int(FLAGS.input_fps/FLAGS.seq_fps)
    seq_length = int(FLAGS.input_length / seq_pool)
    num_event_classes = _get_num_classes(FLAGS.label_mode)
    num_classes = num_event_classes + 1
    if FLAGS.model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(FLAGS.input_length, num_classes, L2_LAMBDA)
    elif FLAGS.model == "inert_small_cnn_lstm":
        model = inert_small_cnn_lstm.Model(num_classes, L2_LAMBDA)
    elif FLAGS.model == "inert_heydarian_cnn_lstm":
        model = inert_heydarian_cnn_lstm.Model(num_classes, L2_LAMBDA)
    # Load weights
    model.load_weights(os.path.join(FLAGS.model_dir, "checkpoints", FLAGS.model_ckpt))
    # Set up metrics
    cl_metrics = {}
    for i in range(1, num_event_classes + 1):
        cl_metrics["tp_{}".format(i)] = 0; cl_metrics["fp1_{}".format(i)] = 0
        cl_metrics["fp2_{}".format(i)] = 0; cl_metrics["fp3_{}".format(i)] = 0
        cl_metrics["fn_{}".format(i)] = 0
    # Files for predicting
    filenames = gfile.Glob(os.path.join(FLAGS.eval_dir, "*.tfrecord"))
    # For each filename, export logits
    for filename in filenames:
        logging.info("Working on {0}.".format(filename))
        # Get the dataset
        data = dataset(is_training=False, is_predicting=True, data_dir=filename)
        # Iterate through batches
        for i, (b_features, b_labels) in enumerate(data):
            # Adjust labels
            b_labels = _adjust_labels(b_labels, seq_pool,
                FLAGS.input_length, FLAGS.batch_size)
            # Run the forward pass
            b_logits = model(b_features, training=False)
            # Collect labels and logits
            labels = b_labels[0] if i==0 else tf.concat([labels, b_labels[:, -1]], 0)
            logits = b_logits[0] if i==0 else tf.concat([logits, b_logits[:, -1]], 0)
            # Collect predictions
            if 'batch_level' in FLAGS.predict_mode:
                # Decode on batch level
                b_preds, _ = decode(b_logits,
                    loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                    blank_index=BLANK_INDEX, def_val=DEF_VAL)
                if FLAGS.predict_mode == 'batch_level_voted':
                    # Construct preds tensor
                    if i == 0:
                        preds = tf.zeros([seq_length, num_classes], tf.int32)
                    else:
                        preds = tf.concat([preds, tf.zeros([1, num_classes], tf.int32)], 0)
                    # Add votes
                    preds_incr = tf.concat([
                        tf.zeros([i, num_classes], tf.int32),
                        tf.one_hot(b_preds[0], depth=num_classes, dtype=tf.int32)], 0)
                    preds += preds_incr
                else:
                    preds = b_preds[0] if i==0 else tf.concat([preds, b_preds[:, -1]], 0)
        # Video level prediction
        v_seq_length = logits.get_shape()[0]
        if FLAGS.predict_mode == 'video_level':
            # Decode on video level
            preds, _ = decode(tf.expand_dims(logits, 0),
                loss_mode=FLAGS.loss_mode, seq_length=v_seq_length,
                blank_index=BLANK_INDEX, def_val=DEF_VAL)
        elif FLAGS.predict_mode == 'batch_level':
            preds = tf.expand_dims(preds, 0)
        elif FLAGS.predict_mode == 'batch_level_voted':
            # Evaluate votes
            preds = tf.argmax(preds, 1, tf.int32)
            preds = tf.expand_dims(preds, 0)
        # Collapse on video level
        preds = collapse(preds, seq_length=v_seq_length, def_val=DEF_VAL,
            pad_val=PAD_VAL)
        # Update metrics
        for i in range(1, num_event_classes + 1):
            other_vals = [j for j in range(1, num_event_classes + 1) if j != i]
            tp, fp1, fp2, fp3, fn = metrics.evaluate_interval_detection(
                labels=tf.expand_dims(labels, 0), predictions=preds,
                event_val=i, def_val=DEF_VAL, seq_length=v_seq_length,
                other_vals=other_vals)
            fp3 = tf.reduce_sum(fp3)
            cl_metrics["tp_{}".format(i)] += tp
            cl_metrics["fp1_{}".format(i)] += fp1
            cl_metrics["fp2_{}".format(i)] += fp2
            cl_metrics["fp3_{}".format(i)] += fp3
            cl_metrics["fn_{}".format(i)] += fn
            pre = tp / (tp + fp1 + fp2 + fp3)
            rec = tp / (tp + fn)
            f1 = 2 * pre * rec / (pre + rec)
            logging.info("Class {}:".format(i))
            logging.info("TP: {0}, FP1: {1}, FP2: {2}, FP3: {3}, FN: {4}".format(
                tp, fp1, fp2, fp3, fn))
            logging.info("Precision: {}, Recall: {}".format(pre, rec))
            logging.info("F1: {0}".format(f1))
        logging.info("===================================================")
        preds = tf.reshape(preds, [-1])
        video_id = os.path.splitext(os.path.basename(filename))[0]
        ids = [video_id] * len(logits)
        logging.info("Writing {0} examples to {1}.csv...".format(len(ids), video_id))
        save_array = np.column_stack((ids, labels.numpy().tolist(),
            logits.numpy().tolist(), preds.numpy().tolist()))
        if not os.path.exists("predict"):
            os.makedirs("predict")
        np.savetxt("predict/{0}.csv".format(video_id), save_array, delimiter=",", fmt='%s')
    # Print metrics
    logging.info("===================== Finished ====================")
    m_pre = 0; m_rec = 0; m_f1 = 0;
    for i in range(1, num_event_classes + 1):
        logging.info("---------------------- Class {} --------------------".format(i))
        cl_tp = cl_metrics["tp_{}".format(i)]
        cl_fp1 = cl_metrics["fp1_{}".format(i)]
        cl_fp2 = cl_metrics["fp2_{}".format(i)]
        cl_fp3 = cl_metrics["fp3_{}".format(i)]
        cl_fn = cl_metrics["fn_{}".format(i)]
        cl_pre = cl_tp / (cl_tp + cl_fp1 + cl_fp2 + cl_fp3)
        cl_rec = cl_tp / (cl_tp + cl_fn)
        cl_f1 = 2 * cl_pre * cl_rec / (cl_pre + cl_rec)
        logging.info("TP: {0}, FP1: {1}, FP2: {2}, FP3: {3}, FN: {4}".format(
            cl_tp, cl_fp1, cl_fp2, cl_fp3, cl_fn))
        logging.info("Precision: {0}, Recall: {1}".format(cl_pre, cl_rec))
        logging.info("F1: {0}".format(cl_f1))
        m_pre += cl_pre; m_rec += cl_rec; m_f1 += cl_f1
    logging.info("===================================================")
    logging.info("mPrecision: {0}, mRecall: {1}".format(
        m_pre/num_event_classes, m_rec/num_event_classes))
    logging.info("mF1: {0}".format(m_f1/num_event_classes))

def dataset(is_training, is_predicting, data_dir):
    """Input pipeline"""
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
    if FLAGS.input_mode == 'inert':
        table = _get_hash_table(FLAGS.label_mode)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_TRAINING_FILES)
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=_get_input_parser(table),
                num_parallel_calls=AUTOTUNE)
            .apply(_get_sequence_batch_fn(is_training, is_predicting))
            #.filter(predicate=_get_def_label_filter(is_training))
            .map(map_func=_get_transformation_parser(is_training),
                num_parallel_calls=AUTOTUNE),
        cycle_length=4)
    if is_training:
        dataset = dataset.shuffle(NUM_SHUFFLE)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

class Adam(keras.optimizers.Adam):
    """Adam optimizer that retrieves learning rate based on epochs"""
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self._epochs = None
    def _decayed_lr(self, var_dtype):
        """Get learning rate based on epochs."""
        lr_t = self._get_hyper("learning_rate", var_dtype)
        if isinstance(lr_t, keras.optimizers.schedules.LearningRateSchedule):
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

def _adjust_labels(labels, seq_pool, seq_length, batch_size=None):
    """If seq_pool performed, adjust seq_length and labels by slicing"""
    if batch_size is not None:
        if seq_pool > 1:
            labels = tf.strided_slice(
                input_=labels, begin=[0, seq_pool-1], end=[batch_size, seq_length],
                strides=[1, seq_pool])
        labels = tf.reshape(labels, [batch_size, int(seq_length/seq_pool)])
    else:
        if seq_pool > 1:
            labels = tf.strided_slice(
                input_=labels, begin=[seq_pool-1], end=[seq_length],
                strides=[seq_pool])
        labels = tf.reshape(labels, [int(seq_length/seq_pool)])
    return labels

def _get_input_parser(table):
    """Return the input parser"""

    def input_parser_video(serialized_example):
        """Parser for raw video"""
        features = tf.io.parse_single_example(
            serialized_example, {
                'example/{}'.format(FLAGS.label_mode): tf.io.FixedLenFeature([], dtype=tf.int64),
                'example/image': tf.io.FixedLenFeature([], dtype=tf.string)
        })
        label = tf.cast(features['example/{}'.format(FLAGS.label_mode)], tf.int32)
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
                'example/{}'.format(FLAGS.label_mode): tf.io.FixedLenFeature([], dtype=tf.string),
                'example/dom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/dom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_acc': tf.io.FixedLenFeature([3], dtype=tf.float32),
                'example/ndom_gyro': tf.io.FixedLenFeature([3], dtype=tf.float32)
        })
        label = tf.cast(table.lookup(features['example/{}'.format(FLAGS.label_mode)]), tf.int32)
        features = tf.stack(
            [features['example/dom_acc'], features['example/dom_gyro'],
             features['example/ndom_acc'], features['example/ndom_gyro']], 0)
        features = tf.squeeze(tf.reshape(features, [-1, 12]))
        return features, label

    if FLAGS.input_mode == "video":
        return input_parser_video
    elif FLAGS.input_mode == "inert":
        return input_parser_inert

def _get_sequence_batch_fn(is_training, is_predicting):
    """Return sliding batched dataset or batched dataset."""
    if is_training or is_predicting:
        shift = int(FLAGS.input_fps/FLAGS.seq_fps)
    else:
        shift = FLAGS.input_length
    return lambda dataset: dataset.window(
        size=FLAGS.input_length, shift=shift, drop_remainder=True).flat_map(
            lambda f_w, l_w: tf.data.Dataset.zip(
                (f_w.batch(FLAGS.input_length), l_w.batch(FLAGS.input_length))))

def _get_transformation_parser(is_training):
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
            size = [FLAGS.input_length, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
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

    if FLAGS.input_mode == "video":
        return image_transformation_parser
    elif FLAGS.input_mode == "inert":
        return inert_transformation_parser

def _get_def_label_filter(is_training):
    if is_training:
        def predicate_fn(features, labels):
            seq_pool = int(FLAGS.input_fps/FLAGS.seq_fps)
            labels = _adjust_labels(labels, seq_pool, FLAGS.input_length)
            return tf.math.greater(tf.reduce_max(labels), 0)
        return predicate_fn
    else:
        def predicate_fn(features, labels):
            return tf.math.greater_equal(tf.reduce_max(labels), 0)
        return predicate_fn

def _get_num_classes(label_category):
    return NUM_EVENT_CLASSES_MAP[label_category]

def _get_hash_table(label_category):
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

def main(arg=None):
    if FLAGS.mode == 'train_and_evaluate':
        train_and_evaluate()
    elif FLAGS.mode == 'predict':
        predict()

# Run
if __name__ == "__main__":
    app.run(main=main)
