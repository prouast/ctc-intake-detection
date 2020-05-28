import os
import gc
import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow import keras
from ctc import decode
from ctc import loss
from ctc import collapse
from model_saver import ModelSaver
import metrics
import oreba_dis
import fic
import clemson
import video_small_cnn_lstm
import video_resnet_cnn_lstm
import inert_small_cnn_lstm
import inert_heyd_cnn_lstm
import inert_resnet_cnn_lstm

# Representation
# Event class vals will be consecutive numbers after DEF_VAL
BLANK_INDEX = 0
DEF_VAL = 1
PAD_VAL = 0
USE_DEF = True

# Hyperparameters
L2_LAMBDA = 1e-5
LR_BOUNDARIES = [5, 10, 15]
LR_VALUE_DIV = [1., 10., 100., 1000.]
LR_DECAY_RATE = 0.8
LR_DECAY_STEPS = 1
NUM_SHUFFLE = 100000

LABEL_MODES = oreba_dis.EVENT_NAMES_MAP.keys()

FLAGS = flags.FLAGS
flags.DEFINE_integer(name='batch_size',
    default=128, help='Batch size used for training.')
flags.DEFINE_enum(name='dataset',
    default='oreba-dis', enum_values=["oreba-dis", "fic", "clemson"],
    help='Select the dataset')
flags.DEFINE_string(name='eval_dir',
    default='data/inert/eval', help='Directory for val data.')
flags.DEFINE_integer(name='eval_steps',
    default=250, help='Eval and save best model after every x steps.')
flags.DEFINE_integer(name='input_length',
    default=128, help='Number of input sequence elements.')
flags.DEFINE_enum(name='input_mode',
    default="inert", enum_values=["video", "inert"],
    help='What is the input mode')
flags.DEFINE_enum(name='label_mode',
    default="label_1", enum_values=LABEL_MODES,
    help='What is the label mode')
flags.DEFINE_integer(name='log_steps',
    default=50, help='Log after every x steps.')
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
    default='inert_small_cnn_lstm',
    enum_values=["video_small_cnn_lstm", "video_resnet_cnn_lstm",
        "inert_small_cnn_lstm", "inert_heyd_cnn_lstm", "inert_resnet_cnn_lstm"],
    help='Select the model')
flags.DEFINE_string(name='model_ckpt',
    default=None, help='Model checkpoint for prediction (e.g., model_5000).')
flags.DEFINE_string(name='model_dir',
    default='run', help='Output directory for model and training stats.')
flags.DEFINE_enum(name='predict_mode',
    default='batch_level_voted', enum_values=['video_level', 'batch_level', 'batch_level_voted'],
    help='How should the predictions be aggregated?')
flags.DEFINE_boolean(name='profile', default='False', help="Save profile")
flags.DEFINE_integer(name='seq_shift',
    default=1, help='Shift when generating sequences.')
flags.DEFINE_string(name='train_dir',
    default='data/inert/train', help='Directory for training data.')
flags.DEFINE_integer(name='train_epochs',
    default=120, help='Number of training epochs.')

logging.set_verbosity(logging.INFO)

def _get_dataset(dataset, label_mode, input_mode, input_length, seq_shift, def_val):
    """Get the dataset"""
    if dataset == 'oreba-dis':
        dataset = oreba_dis.Dataset(label_mode, input_mode, input_length,
            seq_shift, def_val)
    elif dataset == 'fic':
        dataset = fic.Dataset(label_mode, input_length, seq_shift, def_val)
    elif dataset == 'clemson':
        dataset = clemson.Dataset(label_mode, input_length, seq_shift, def_val)
    else:
        raise ValueError("Dataset {} not implemented!".format(FLAGS.dataset))

    return dataset

def _get_model(model, dataset, num_classes, input_length, l2_lambda):
    """Get the model"""
    if model == "video_small_cnn_lstm":
        model = video_small_cnn_lstm.Model(num_classes=num_classes,
            input_length=input_length, l2_lambda=l2_lambda)
    elif model == "video_resnet_cnn_lstm":
        model = video_resnet_cnn_lstm.Model(num_classes=num_classes,
            input_length=input_length, l2_lambda=l2_lambda)
    elif model == "inert_small_cnn_lstm":
        if dataset == "oreba-dis":
            specs = {
                "seq_pool": 8,
                "num_conv": [(64, 7, True), (128, 5, True), (256, 3, True)],
                "num_lstm": [64]
            }
        elif dataset == "clemson":
            specs = {
                "seq_pool": 4,
                "num_conv": [(64, 7, False), (128, 5, True), (256, 3, True)],
                "num_lstm": [64]
            }
        elif dataset == "fic":
            raise ValueError("Model not implemented for {}!".format(FLAGS.dataset))
        model = inert_small_cnn_lstm.Model(num_classes=num_classes,
            input_length=input_length, specs=specs, l2_lambda=l2_lambda)
    elif model == "inert_heyd_cnn_lstm":
        # Model does not need to be customized per dataset since it does
        #  no sequence pooling
        model = inert_heyd_cnn_lstm.Model(num_classes=num_classes,
            input_length=input_length, l2_lambda=l2_lambda)
    elif model == "inert_resnet_cnn_lstm":
        if dataset == "oreba-dis":
            specs = {
                "seq_pool": 8,
                "conv_1_filters": 64,
                "conv_1_kernel_size": 7,
                "block_specs": [(2, 64, 5, 1), (2, 128, 5, 2), (2, 256, 3, 2),
                    (2, 512, 3, 2)],
                "lstm_specs": [(64, False), (64, True)]
            }
        elif dataset == "clemson":
            specs = {
                "seq_pool": 4,
                "conv_1_filters": 64,
                "conv_1_kernel_size": 7,
                "block_specs": [(2, 64, 5, 1), (2, 128, 5, 2), (2, 256, 3, 1),
                    (2, 512, 3, 2)],
                "lstm_specs": [(64, False), (64, True)]
            }
        elif dataset == "fic":
            raise ValueError("Model not implemented for {}!".format(FLAGS.dataset))
        model = inert_resnet_cnn_lstm.Model(num_classes=num_classes,
            input_length=input_length, specs=specs, l2_lambda=l2_lambda)
    else:
        raise ValueError("Model not implemented for {}!".format(model))
    return model

def train_and_evaluate():
    """Run the experiment."""

    # Get dataset
    dataset = _get_dataset(dataset=FLAGS.dataset, label_mode=FLAGS.label_mode,
        input_mode=FLAGS.input_mode, input_length=FLAGS.input_length,
        seq_shift=FLAGS.seq_shift, def_val=DEF_VAL)

    # Read the representation choice
    num_event_classes = dataset.num_classes() # Number of classes including idle
    num_def_classes = 1 if USE_DEF else 0 # Number of classes including DEF
    num_classes = num_event_classes + num_def_classes + 1 # Total number of classes
    shift = DEF_VAL - 1 if USE_DEF else DEF_VAL #

    # Read the model choice
    model = _get_model(model=FLAGS.model, dataset=FLAGS.dataset,
        num_classes=num_classes, input_length=FLAGS.input_length,
        l2_lambda=L2_LAMBDA)

    # Load weights
    if FLAGS.model_ckpt is not None:
        logging.info("Loading from {}".format(FLAGS.model_ckpt))
        model.load_weights(os.path.join(FLAGS.model_dir, "checkpoints", FLAGS.model_ckpt))

    # Get the seq_length
    seq_length = model.seq_length()

    # Instantiate learning rate schedule and optimizer
    if FLAGS.lr_decay_fn == "exponential":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=FLAGS.lr_base,
            decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE, staircase=True)
    elif FLAGS.lr_decay_fn == "piecewise_constant":
        values = np.divide(FLAGS.lr_base, LR_VALUE_DIV)
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=LR_BOUNDARIES, values=values.tolist())
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=10.0)

    # Get train and eval dataset
    train_dataset = dataset(batch_size=FLAGS.batch_size, is_training=True,
        is_predicting=False, data_dir=FLAGS.train_dir, num_shuffle=NUM_SHUFFLE)
    eval_dataset = dataset(batch_size=FLAGS.batch_size, is_training=False,
        is_predicting=False, data_dir=FLAGS.eval_dir, num_shuffle=NUM_SHUFFLE)

    # Instantiate the metrics
    train_metrics = {
        'mean_precision': keras.metrics.Mean(),
        'mean_recall': keras.metrics.Mean(),
        'mean_f1': keras.metrics.Mean()}
    eval_metrics = {
        'mean_precision': keras.metrics.Mean(),
        'mean_recall': keras.metrics.Mean(),
        'mean_f1': keras.metrics.Mean()}
    # event_classes are all classes except DEF
    event_classes = range(DEF_VAL + 1, DEF_VAL + num_event_classes + 1)
    for i in event_classes:
        other_vals = [j for j in event_classes if j != i]
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

            # Start profiling
            if FLAGS.profile:
                tf.profiler.experimental.start('run/profiler')

            # Adjust labels as specified by model
            train_labels = model.labels(train_labels, batch_size=FLAGS.batch_size)

            @tf.function
            def _train_step():
                # Open a GradientTape to record the operations run during forward pass
                with tf.GradientTape() as tape:
                    # Run the forward pass
                    train_logits = model(train_features, training=True)
                    # The loss function
                    train_loss = loss(train_labels, train_logits,
                        loss_mode=FLAGS.loss_mode, batch_size=FLAGS.batch_size,
                        seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL,
                        blank_index=BLANK_INDEX, training=True, use_def=USE_DEF)
                    # l2 regularization loss
                    l2_loss = sum(model.losses)
                    # Gradients
                    grads = tape.gradient(train_loss+l2_loss, model.trainable_weights)
                    # Apply the gradients
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    return train_logits, train_loss, grads, l2_loss

            train_logits, train_loss, grads, l2_loss = _train_step()

            # Decode logits into predictions
            train_predictions_u, train_predictions = decode(train_logits,
                loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                blank_index=BLANK_INDEX, def_val=DEF_VAL, use_def=USE_DEF,
                shift=shift)
            train_predictions_u = collapse(train_predictions_u,
                seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL)

            # Stop profiling
            if FLAGS.profile:
                tf.profiler.experimental.stop()

            # Log every FLAGS.log_steps steps.
            if global_step % FLAGS.log_steps == 0:
                # Update metrics
                for i in event_classes:
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
                    tf.summary.scalar("training/global_gradient_norm",
                        data=tf.linalg.global_norm(grads), step=global_step)
                    tf.summary.scalar('training/loss', data=train_loss, step=global_step)
                    tf.summary.scalar('training/l2_loss', data=l2_loss, step=global_step)
                    tf.summary.scalar('training/total_loss', data=train_loss+l2_loss, step=global_step)
                    tf.summary.scalar('training/learning_rate', data=lr_schedule(epoch), step=global_step)
                    tf.summary.scalar('metrics/mean_precision', data=train_metrics['mean_precision'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_recall', data=train_metrics['mean_recall'].result(), step=global_step)
                    tf.summary.scalar('metrics/mean_f1', data=train_metrics['mean_f1'].result(), step=global_step)
                # Reset metrics
                train_metrics['mean_precision'].reset_states()
                train_metrics['mean_recall'].reset_states()
                train_metrics['mean_f1'].reset_states()
                # For each class
                for i in event_classes:
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

                    # Adjust labels as specified by model
                    eval_labels = model.labels(eval_labels,
                        batch_size=FLAGS.batch_size)

                    @tf.function
                    def _eval_step():
                        # Run the forward pass
                        eval_logits = model(eval_features, training=False)
                        # The loss function
                        eval_loss = loss(eval_labels, eval_logits,
                            loss_mode=FLAGS.loss_mode, batch_size=FLAGS.batch_size,
                            seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL,
                            blank_index=BLANK_INDEX, training=False, use_def=USE_DEF)
                        return eval_logits, eval_loss

                    eval_logits, eval_loss = _eval_step()
                    eval_losses.append(eval_loss.numpy())

                    # Decode logits into predictions
                    eval_predictions_u, eval_predictions = decode(eval_logits,
                        loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                        blank_index=BLANK_INDEX, def_val=DEF_VAL,
                        use_def=USE_DEF, shift=shift)
                    eval_predictions_u = collapse(eval_predictions_u,
                        seq_length=seq_length, def_val=DEF_VAL, pad_val=PAD_VAL)

                    # Update metric
                    for i in event_classes:
                        eval_metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_precision'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_recall'.format(i)](eval_labels, eval_predictions_u)
                        eval_metrics['class_{}_f1'.format(i)](eval_labels, eval_predictions_u)

                for i in event_classes:
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
                for i in event_classes:
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

            # Clean up memory
            tf.keras.backend.clear_session()
            gc.collect()

            # Increment global step
            global_step += 1

        logging.info('Finished epoch %s' % (epoch,))
        optimizer.finish_epoch()

def predict():
    assert FLAGS.batch_size == 1, "batch_size should be 1 for prediction"
    # Get dataset
    dataset = _get_dataset(dataset=FLAGS.dataset, label_mode=FLAGS.label_mode,
        input_mode=FLAGS.input_mode, input_length=FLAGS.input_length,
        seq_shift=FLAGS.seq_shift, def_val=DEF_VAL)
    # Get the representation
    num_event_classes = dataset.num_classes()
    num_def_classes = 1 if USE_DEF else 0
    num_classes = num_event_classes + num_def_classes + 1
    shift = DEF_VAL - 1 if USE_DEF else DEF_VAL
    # Read the model choice
    model = _get_model(model=FLAGS.model, dataset=FLAGS.dataset,
        num_classes=num_classes, input_length=FLAGS.input_length,
        l2_lambda=L2_LAMBDA)
    # Make sure that seq_shift is set corresponding to model SEQ_POOL
    assert FLAGS.seq_shift == model.seq_pool(), \
        "seq_shift should be equal to model.seq_pool() in predict"
    # Get the seq length
    seq_length = model.seq_length()
    # Load weights
    model.load_weights(os.path.join(FLAGS.model_dir, "checkpoints", FLAGS.model_ckpt))
    # Set up metrics
    cl_metrics = {}
    event_classes = range(DEF_VAL + 1, DEF_VAL + num_event_classes + 1)
    for i in event_classes:
        cl_metrics["tp_{}".format(i)] = 0; cl_metrics["fp1_{}".format(i)] = 0
        cl_metrics["fp2_{}".format(i)] = 0; cl_metrics["fp3_{}".format(i)] = 0
        cl_metrics["fn_{}".format(i)] = 0
    # Files for predicting
    filenames = gfile.Glob(os.path.join(FLAGS.eval_dir, "*.tfrecord"))
    # For each filename, export logits
    for filename in filenames:
        logging.info("Working on {0}.".format(filename))
        # Get the dataset
        data = dataset(batch_size=FLAGS.batch_size, is_training=False,
            is_predicting=True, data_dir=filename)
        # Iterate through batches
        for i, (b_features, b_labels) in enumerate(data):
            # Adjust labels as specified by model
            b_labels = model.labels(b_labels, batch_size=FLAGS.batch_size)
            @tf.function
            def _pred_step():
                # Run the forward pass
                b_logits = model(b_features, training=False)
                return b_logits
            b_logits = _pred_step()
            # Collect labels and logits
            labels = b_labels[0] if i==0 else tf.concat([labels, b_labels[:, -1]], 0)
            logits = b_logits[0] if i==0 else tf.concat([logits, b_logits[:, -1]], 0)
            # Collect predictions
            if 'batch_level' in FLAGS.predict_mode:
                # Decode on batch level
                b_preds, _ = decode(b_logits,
                    loss_mode=FLAGS.loss_mode, seq_length=seq_length,
                    blank_index=BLANK_INDEX, def_val=DEF_VAL, use_def=USE_DEF,
                    shift=shift)
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
        v_seq_length = tf.constant(logits.get_shape()[0], tf.int64)
        if FLAGS.predict_mode == 'video_level':
            # Decode on video level
            preds, _ = decode(tf.expand_dims(logits, 0),
                loss_mode=FLAGS.loss_mode, seq_length=v_seq_length,
                blank_index=BLANK_INDEX, def_val=DEF_VAL, use_def=USE_DEF,
                shift=shift)
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
        for i in event_classes:
            other_vals = [j for j in event_classes if j != i]
            other_vals = tf.constant(other_vals, tf.int32)
            tp, fp1, fp2, fp3, fn = metrics.evaluate_interval_detection(
                labels=tf.expand_dims(labels, 0), predictions=preds,
                event_val=tf.constant(i), def_val=tf.constant(DEF_VAL),
                seq_length=v_seq_length, other_vals=other_vals)
            fp3 = tf.reduce_sum(fp3)
            cl_metrics["tp_{}".format(i)] += tp
            cl_metrics["fp1_{}".format(i)] += fp1
            cl_metrics["fp2_{}".format(i)] += fp2
            cl_metrics["fp3_{}".format(i)] += fp3
            cl_metrics["fn_{}".format(i)] += fn
            pre = tp / (tp + fp1 + fp2 + fp3)
            rec = tp / (tp + fn)
            f1 = 2 * pre * rec / (pre + rec)
            logging.info("---------------------- Class {} --------------------".format(i))
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
    for i in event_classes:
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

def main(arg=None):
    if FLAGS.mode == 'train_and_evaluate':
        train_and_evaluate()
    elif FLAGS.mode == 'predict':
        predict()

# Run
if __name__ == "__main__":
    app.run(main=main)
