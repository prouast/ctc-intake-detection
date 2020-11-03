import os
import gc
import numpy as np
import json
import psutil
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.platform import gfile
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from representation import Representation
from model_saver import ModelSaver
from metrics import TrainMetrics
from metrics import EvalMetrics
from metrics import PredMetrics
import aggregation
import oreba_dis
import fic
import clemson
import video_resnet_cnn_lstm
import inert_resnet_cnn_lstm
import inert_heydarian_cnn_lstm
import inert_kyritsis_cnn_lstm

# Representation
# Event class vals will be consecutive numbers after DEF_VAL
BLANK_INDEX = 0
DEF_VAL = 1
PAD_VAL = 0

# Hyperparameters
L2_LAMBDA = 1e-5
LR_BOUNDARIES = [5, 10, 15]
LR_VALUE_DIV = [1., 10., 100., 1000.]
LR_DECAY_STEPS = 1
LABEL_MODES = clemson.EVENT_NAMES_MAP.keys()

FLAGS = flags.FLAGS
flags.DEFINE_integer(name='batch_size',
  default=128, help='Batch size used for training. Predict uses 1.')
flags.DEFINE_integer(name='beam_width',
  default=10, help='Width used for beam search.')
flags.DEFINE_enum(name='dataset',
  default='oreba-dis', enum_values=["oreba-dis", "fic", "clemson"],
  help='Select the dataset')
flags.DEFINE_enum(name='decode_fn',
  default='beam_search', enum_values=["greedy", "beam_search"],
  help='Select the decode fn')
flags.DEFINE_integer(name='eval_batch_size',
  default=1, help='Batch size used for evaluation. Predict uses 1.')
flags.DEFINE_string(name='eval_dir',
  default='data/inert/valid', help='Directory for val data.')
flags.DEFINE_integer(name='eval_steps',
  default=1000, help='Eval and save best model after every x steps.')
flags.DEFINE_integer(name='input_length',
  default=128, help='Number of input sequence elements.')
flags.DEFINE_enum(name='input_mode',
  default="inert", enum_values=["video", "inert"],
  help='What is the input mode')
flags.DEFINE_enum(name='label_mode',
  default="label_1", enum_values=LABEL_MODES,
  help='What is the label mode')
flags.DEFINE_integer(name='log_steps',
  default=250, help='Log after every x steps.')
flags.DEFINE_enum(name='loss_mode',
  default="ctc", enum_values=["ctc", "crossent"], help="What is the loss mode")
flags.DEFINE_float(name='lr_base',
  default=1e-3, help='Base learning rate.')
flags.DEFINE_enum(name='lr_decay_fn',
  default="exponential", enum_values=["constant", "exponential", "piecewise_constant"],
  help='How does learning rate decay')
flags.DEFINE_float(name='lr_decay_rate',
  default=0.9, help='Decay for exponential learning rate.')
flags.DEFINE_boolean(name='mixed_precision',
  default=False,   help="Use mixed precision")
flags.DEFINE_enum(name='mode',
  default="train_and_evaluate", enum_values=["train_and_evaluate", "predict"],
  help='What mode should tensorflow be started in')
flags.DEFINE_enum(name='model',
  default='inert_resnet_cnn_lstm',
  enum_values=["video_resnet_cnn_lstm", "inert_resnet_cnn_lstm", "inert_kyritsis_cnn_lstm", "inert_heydarian_cnn_lstm"],
  help='Select the model')
flags.DEFINE_string(name='model_ckpt',
  default=None, help='Model checkpoint for prediction (e.g., model_5000).')
flags.DEFINE_string(name='model_dir',
  default='run', help='Output directory for model and training stats.')
flags.DEFINE_integer(name='num_shuffle',
  default=50000, help='Size of the shuffle buffer.')
flags.DEFINE_string(name='predict_dir',
  default='predict', help='Output directory for prediction mode.')
flags.DEFINE_enum(name='predict_mode',
  default='batch_level_voted',
  enum_values=['video_level_average', 'video_level_concat',
    'batch_level_concat', 'batch_level_voted', 'probs'],
  help='How should the predictions be aggregated?')
flags.DEFINE_integer(name='seq_shift',
  default=2, help='Shift when generating sequences.')
flags.DEFINE_string(name='train_dir',
  default='data/inert/train', help='Directory for training data.')
flags.DEFINE_integer(name='train_epochs',
  default=60, help='Number of training epochs.')
flags.DEFINE_boolean(name='use_def',
  default='True', help="Use def class for representation")

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
  if model == "video_resnet_cnn_lstm":
    model = video_resnet_cnn_lstm.Model(num_classes=num_classes,
      input_length=input_length, l2_lambda=l2_lambda)
  elif model == "inert_resnet_cnn_lstm":
    if dataset == "oreba-dis":
      specs = {
        "seq_pool": 8,
        "conv_1_filters": 64,
        "conv_1_kernel_size": 1,
        "conv_1_stride": 1,
        "block_specs": [(1, 64, 3, 1), (1, 128, 3, 2), (1, 256, 5, 2),
          (1, 512, 5, 2)],
        "lstm_specs": [(64, False)]
      }
    elif dataset == "clemson":
      specs = {
        "seq_pool": 2,
        "conv_1_filters": 64,
        "conv_1_kernel_size": 1,
        "conv_1_stride": 1,
        "block_specs": [(1, 64, 3, 1), (1, 128, 3, 1), (1, 256, 5, 2),
          (1, 512, 5, 1)],
        "lstm_specs": [(64, False)]
      }
    elif dataset == "fic":
      specs = {
        "seq_pool": 16,
        "conv_1_filters": 64,
        "conv_1_kernel_size": 1,
        "conv_1_stride": 2,
        "block_specs": [(1, 64, 3, 1), (1, 128, 3, 2), (1, 256, 5, 2),
          (1, 512, 5, 2)],
        "lstm_specs": [(64, False)]
      }
    model = inert_resnet_cnn_lstm.Model(num_classes=num_classes,
      input_length=input_length, specs=specs, l2_lambda=l2_lambda)
  elif model == "inert_kyritsis_cnn_lstm":
    if dataset == "oreba-dis":
      specs = {
        "seq_pool": 4,
        "conv_layer_specs": [(64, 6, True), (128, 6, True)]
      }
    elif dataset == "clemson":
      specs = {
        "seq_pool": 1,
        "conv_layer_specs": [(64, 3, False), (128, 3, False)]
      }
    model = inert_kyritsis_cnn_lstm.Model(num_classes=num_classes,
      input_length=input_length, specs=specs, l2_lambda=l2_lambda)
  elif model == "inert_heydarian_cnn_lstm":
    model = inert_heydarian_cnn_lstm.Model(num_classes=num_classes,
      input_length=input_length, l2_lambda=l2_lambda)
  else:
    raise ValueError("Model not implemented for {}!".format(model))
  return model

def _get_preds_aggregator(predict_mode, n, rep, v_seq_length):
  seq_length = rep.get_seq_length()
  num_classes = rep.get_num_classes()
  if FLAGS.predict_mode == "video_level_concat":
    # Concat last logit each step and decode on video level
    decode_fn = rep.get_decode_fn(
      batch_size=1, seq_length=v_seq_length)
    logits_aggregator = aggregation.ConcatAggregator(n=n, idx=seq_length-1)
    preds_aggregator = aggregation.VideoLevelPredsAggregator(
      logits_aggregator=logits_aggregator, decode_fn=decode_fn)
  elif FLAGS.predict_mode == "video_level_average":
    # Average logits across steps and decode on video level
    decode_fn = rep.get_decode_fn(
      batch_size=1, seq_length=v_seq_length)
    logits_aggregator = aggregation.AverageAggregator(
      num_classes=num_classes, seq_length=seq_length)
    preds_aggregator = aggregation.VideoLevelPredsAggregator(
      logits_aggregator=logits_aggregator, decode_fn=decode_fn)
  elif FLAGS.predict_mode == "batch_level_concat":
    # Decode logits on batch level and concat preds across steps
    decode_fn = rep.get_decode_fn(1)
    preds_aggregator = aggregation.BatchLevelConcatPredsAggregator(
      n=n, idx=tf.math.floordiv(seq_length, 2), decode_fn=decode_fn)
  elif FLAGS.predict_mode == "batch_level_voted":
    # Decode logits on batch level and vote preds across steps
    decode_fn = rep.get_decode_fn(1)
    preds_aggregator = aggregation.BatchLevelVotedPredsAggregator(
      num_classes=num_classes, seq_length=seq_length, def_val=DEF_VAL,
      decode_fn=decode_fn)
  else:
    logging.info("No preds aggregator selected.")
    preds_aggregator = None
  return preds_aggregator

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

@tf.function
def train_step(model, train_features, train_labels, train_labels_c, train_labels_l, loss_fn, optimizer):
  # Open a GradientTape to record the operations run during forward pass
  with tf.GradientTape() as tape:
    # Run the forward pass
    train_logits = model(train_features, training=True)
    # The loss function
    train_loss = loss_fn(train_labels, train_labels_c, train_logits, train_labels_l)
    # l2 regularization loss
    train_l2_loss = sum(model.losses)
    if FLAGS.mixed_precision:
      # Get scaled loss
      scaled_train_loss = optimizer.get_scaled_loss(train_loss)
      scaled_train_l2_loss = optimizer.get_scaled_loss(train_l2_loss)
      # Gradients
      scaled_train_grads = tape.gradient(
        scaled_train_loss+scaled_train_l2_loss, model.trainable_weights)
      # Get unscaled gradients
      train_grads = optimizer.get_unscaled_gradients(scaled_train_grads)
    else:
      # Gradients
      train_grads = tape.gradient(
        train_loss+train_l2_loss, model.trainable_weights)
  # Apply the gradients
  optimizer.apply_gradients(zip(train_grads, model.trainable_weights))
  return train_logits, train_loss, train_l2_loss, train_grads

@tf.function
def eval_step(model, eval_features, eval_labels, eval_labels_c, eval_labels_l, loss_fn):
  # Run the forward pass
  eval_logits = model(eval_features, training=False)
  # The loss function
  eval_loss = loss_fn(eval_labels, eval_labels_c, eval_logits, eval_labels_l)
  # l2 regularization loss
  eval_l2_loss = sum(model.losses)
  return eval_logits, eval_loss, eval_l2_loss

@tf.function
def pred_step(model, b_features):
  # Run the forward pass
  b_logits = model(b_features, training=False)
  return b_logits

def train_and_evaluate():
  """Train the model with custom training loop, evaluating at given intervals."""

  # Set mixed precision policy
  if FLAGS.mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

  # Get dataset
  dataset = _get_dataset(dataset=FLAGS.dataset, label_mode=FLAGS.label_mode,
    input_mode=FLAGS.input_mode, input_length=FLAGS.input_length,
    seq_shift=FLAGS.seq_shift, def_val=DEF_VAL)

  # Define representation
  rep = Representation(blank_index=BLANK_INDEX, def_val=DEF_VAL,
    loss_mode=FLAGS.loss_mode, num_event_classes=dataset.num_event_classes(),
    pad_val=PAD_VAL, use_def=FLAGS.use_def, decode_fn=FLAGS.decode_fn,
    beam_width=FLAGS.beam_width)

  # Get model
  model = _get_model(model=FLAGS.model, dataset=FLAGS.dataset,
    num_classes=rep.get_num_classes(), input_length=FLAGS.input_length,
    l2_lambda=L2_LAMBDA)
  seq_length = model.get_seq_length()
  rep.set_seq_length(seq_length)

  # Instantiate learning rate schedule and optimizer
  if FLAGS.lr_decay_fn == "exponential":
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=FLAGS.lr_base,
      decay_steps=LR_DECAY_STEPS, decay_rate=FLAGS.lr_decay_rate, staircase=True)
  elif FLAGS.lr_decay_fn == "piecewise_constant":
    values = np.divide(FLAGS.lr_base, LR_VALUE_DIV)
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=LR_BOUNDARIES, values=values.tolist())
  elif FLAGS.lr_decay_fn == "constant":
    lr_schedule = ConstantLR(FLAGS.lr_base)
  optimizer = Adam(learning_rate=lr_schedule)
  # Get LossScaleOptimizer
  if FLAGS.mixed_precision:
    optimizer = LossScaleOptimizer(optimizer=optimizer, loss_scale='dynamic')

  # Get loss function
  train_loss_fn = rep.get_loss_fn(batch_size=FLAGS.batch_size)
  eval_loss_fn = rep.get_loss_fn(batch_size=FLAGS.eval_batch_size)

  # Get train and eval dataset
  collapse_fn = rep.get_loss_collapse_fn()
  train_dataset = dataset(batch_size=FLAGS.batch_size, data_dir=FLAGS.train_dir,
    is_predicting=False, is_training=True,
    label_fn=model.get_label_fn(FLAGS.batch_size),
    collapse_fn=collapse_fn, num_shuffle=FLAGS.num_shuffle)
  eval_dataset = dataset(batch_size=FLAGS.eval_batch_size,
    data_dir=FLAGS.eval_dir, is_predicting=False, is_training=False,
    label_fn=model.get_label_fn(FLAGS.eval_batch_size),
    collapse_fn=collapse_fn, num_shuffle=FLAGS.num_shuffle)

  # Load model
  if FLAGS.model_ckpt is not None:
    logging.info("Loading model from {}".format(FLAGS.model_ckpt))
    load_status = model.load_weights(
      os.path.join(FLAGS.model_dir, "checkpoints", FLAGS.model_ckpt))
    load_status.assert_consumed()

  # Set up log writer and metrics
  train_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "log/train"))
  eval_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "log/eval"))
  train_metrics = TrainMetrics(representation=rep, writer=train_writer)
  eval_metrics = EvalMetrics(representation=rep, writer=eval_writer)

  # Save best checkpoints in terms of f1
  model_saver = ModelSaver(os.path.join(FLAGS.model_dir, "checkpoints"),
    compare_fn=lambda x,y: x.score > y.score, sort_reverse=True)

  # Keep track of total global step
  global_step = 0

  # Iterate over epochs
  for epoch in range(FLAGS.train_epochs):
    logging.info('Starting epoch %d' % (epoch,))

    # Iterate over training batches
    for step, (train_features, train_labels, train_labels_c, train_labels_l) in enumerate(train_dataset):
      # Assert sizes
      assert train_labels.shape == [FLAGS.batch_size, seq_length], "Labels shape [batch_size, seq_length]"
      # Run the train step
      train_logits, train_loss, train_l2_loss, train_grads = train_step(model,
        train_features, train_labels, train_labels_c, train_labels_l, train_loss_fn, optimizer)
      # Assert sizes
      assert train_logits.shape == [FLAGS.batch_size, seq_length, rep.get_num_classes()], "Logits shape [batch_size, seq_length, num_classes]"
      # Log every FLAGS.log_steps steps.
      if global_step % FLAGS.log_steps == 0:
        logging.info("Memory used: {} GB".format(psutil.virtual_memory().used/2**30))
        # Decode logits into predictions
        train_predictions_u = None
        if FLAGS.loss_mode == "ctc":
          train_predictions_u, _ = rep.get_decode_fn(FLAGS.batch_size)(train_logits)
          train_predictions_u = rep.get_inference_collapse_fn()(train_predictions_u)
        # General logs
        logging.info('Step %s in epoch %s; global step %s' % (step, epoch, global_step))
        logging.info('Seen this epoch: %s samples' % ((step + 1) * FLAGS.batch_size))
        logging.info('Total loss (this step): %s' % float(train_loss+train_l2_loss))
        with train_writer.as_default():
          tf.summary.scalar("training/global_gradient_norm",
            data=tf.linalg.global_norm(train_grads), step=global_step)
          tf.summary.scalar('training/loss', data=train_loss, step=global_step)
          tf.summary.scalar('training/l2_loss', data=train_l2_loss, step=global_step)
          tf.summary.scalar('training/total_loss', data=train_loss+train_l2_loss, step=global_step)
          tf.summary.scalar('training/learning_rate', data=lr_schedule(epoch), step=global_step)
        # Update metrics
        train_metrics.update(train_labels, train_logits, train_predictions_u)
        # Log metrics
        train_metrics.log(global_step)
        # Save latest model
        model_saver.save_latest(model=model, step=global_step, file="model")
        # Flush TensorBoard
        train_writer.flush()

      # Evaluate every FLAGS.eval_steps steps.
      if global_step % FLAGS.eval_steps == 0:
        logging.info('Evaluating at global step %s' % global_step)
        # Keep track of eval losses
        eval_losses = []
        eval_l2_losses = []
        # Iterate through eval batches
        for i, (eval_features, eval_labels, eval_labels_c, eval_labels_l) in enumerate(eval_dataset):
          # Assert sizes
          assert eval_labels.shape == [FLAGS.eval_batch_size, seq_length], "Labels shape [batch_size, seq_length]"
          # Run the eval step
          eval_logits, eval_loss, eval_l2_loss = eval_step(model,
            eval_features, eval_labels, eval_labels_c, eval_labels_l, eval_loss_fn)
          eval_losses.append(eval_loss.numpy())
          eval_l2_losses.append(eval_l2_loss.numpy())
          # Assert sizes
          assert eval_logits.shape == [FLAGS.eval_batch_size, seq_length, rep.get_num_classes()], "Logits shape [batch_size, seq_length, num_classes]"
          # Decode logits into predictions
          eval_predictions_u = None
          if FLAGS.loss_mode == "ctc":
            eval_predictions_u, _ = rep.get_decode_fn(FLAGS.eval_batch_size)(eval_logits)
            eval_predictions_u = rep.get_inference_collapse_fn()(eval_predictions_u)
          # Update metrics for this batch
          eval_metrics.update_i(eval_labels, eval_logits, eval_predictions_u)
        # Update mean metrics
        eval_score = eval_metrics.update()
        # General logs
        eval_loss = np.mean(eval_losses)
        eval_l2_loss = np.mean(eval_l2_losses)
        logging.info('Evaluation loss: %s' % float(eval_loss+eval_l2_loss))
        with eval_writer.as_default():
          tf.summary.scalar('training/loss', data=eval_loss, step=global_step)
          tf.summary.scalar('training/l2_loss', data=eval_l2_loss, step=global_step)
          tf.summary.scalar('training/total_loss', data=eval_loss+eval_l2_loss, step=global_step)
        # Log metrics
        eval_metrics.log(global_step)
        # Save best models
        model_saver.save_best(model=model, score=float(eval_score),
          step=global_step, file="model")
        # Flush TensorBoard
        eval_writer.flush()

      # Clean up memory
      tf.keras.backend.clear_session()
      gc.collect()

      # Increment global step
      global_step += 1

    # Save and keep latest model for every 10th epoch
    if epoch % 10 == 9:
      model_saver.save_keep(model=model, step=global_step, file="model")

    logging.info('Finished epoch %s' % (epoch,))
    optimizer.finish_epoch()

  # Save final model
  model_saver.save_latest(model=model, step=global_step, file="model")
  # Finished training
  logging.info("Finished training")

def predict():
  # Set mixed precision policy
  if FLAGS.mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
  # Make target dir
  if not os.path.exists(FLAGS.predict_dir):
    os.makedirs(FLAGS.predict_dir)
  # Get dataset
  dataset = _get_dataset(dataset=FLAGS.dataset, label_mode=FLAGS.label_mode,
    input_mode=FLAGS.input_mode, input_length=FLAGS.input_length,
    seq_shift=FLAGS.seq_shift, def_val=DEF_VAL)
  num_event_classes = dataset.num_event_classes()
  # Define representation
  rep = Representation(blank_index=BLANK_INDEX, def_val=DEF_VAL,
    loss_mode=FLAGS.loss_mode, num_event_classes=num_event_classes,
    pad_val=PAD_VAL, use_def=FLAGS.use_def, decode_fn=FLAGS.decode_fn,
    beam_width=FLAGS.beam_width)
  num_classes = rep.get_num_classes()
  # Get model and infer seq_length
  model = _get_model(model=FLAGS.model, dataset=FLAGS.dataset,
    num_classes=num_classes, input_length=FLAGS.input_length,
    l2_lambda=L2_LAMBDA)
  seq_length = model.get_seq_length()
  rep.set_seq_length(seq_length)
  # Make sure that seq_shift is set corresponding to model SEQ_POOL
  assert FLAGS.seq_shift == model.get_out_pool(), \
    "seq_shift should be equal to model.get_out_pool() in predict"
  # Load weights
  model.load_weights(os.path.join(FLAGS.model_dir, "checkpoints", FLAGS.model_ckpt))
  # Set up metrics
  metrics = PredMetrics(rep)
  # Files for predicting
  filenames = gfile.Glob(os.path.join(FLAGS.eval_dir, "*.tfrecord"))
  # For each filename, export logits
  for filename in filenames:
    # Get video id
    video_id = os.path.splitext(os.path.basename(filename))[0]
    export_csv = os.path.join(FLAGS.predict_dir, str(video_id) + ".csv")
    export_tfrecord = os.path.join(FLAGS.predict_dir, "logits", str(video_id) + ".tfrecord")
    logging.info("Working on {0}.".format(video_id))
    if os.path.exists(export_csv) and os.path.exists(export_tfrecord):
      logging.info("Export files already exist. Skipping {0}.".format(filename))
      continue
    # Get the dataset
    label_fn = model.get_label_fn(1)
    collapse_fn = rep.get_loss_collapse_fn()
    data = dataset(batch_size=1, data_dir=filename, is_predicting=True,
      is_training=False, label_fn=label_fn, collapse_fn=collapse_fn)
    # Iterate to get n and v_seq_length
    n = len(list(data))
    v_seq_length = n+seq_length-1
    # Get the aggregators
    labels_aggregator = aggregation.ConcatAggregator(n=n, idx=seq_length-1)
    if seq_length == 1:
      logits_aggregator = aggregation.ConcatAggregator(n=n, idx=seq_length-1)
    else:
      logits_aggregator = aggregation.AverageAggregator(num_classes=num_classes, seq_length=seq_length)
    preds_aggregator = _get_preds_aggregator(predict_mode=FLAGS.predict_mode,
      n=n, rep=rep, v_seq_length=v_seq_length)
    # Iterate through batches
    # Write logits and labels to TFRecord for analysis
    if not os.path.exists(os.path.join(FLAGS.predict_dir, "logits")):
      os.makedirs(os.path.join(FLAGS.predict_dir, "logits"))
    with tf.io.TFRecordWriter(export_tfrecord) as tfrecord_writer:
      for i, (b_features, b_labels) in enumerate(data):
        # Assert sizes
        assert b_labels.shape == [1, seq_length], "Labels shape [1, seq_length]"
        # Prediction step
        b_logits = pred_step(model, b_features)
        assert b_logits.shape == [1, seq_length, rep.get_num_classes()], "Logits shape [1, seq_length, num_classes]"
        # Aggregation step
        labels_aggregator.step(i, b_labels)
        logits_aggregator.step(i, b_logits)
        if preds_aggregator is not None:
          preds_aggregator.step(i, b_logits)
        example = tf.train.Example(features=tf.train.Features(feature={
          'example/logits': _floats_feature(b_logits.numpy().ravel()),
          'example/labels': _int64_feature(b_labels.numpy().ravel())
        }))
        tfrecord_writer.write(example.SerializeToString())
    # Get aggregated data
    labels = labels_aggregator.result()
    logits = logits_aggregator.result()
    preds = None
    if preds_aggregator is not None:
      preds = preds_aggregator.result()
    # Collapse on video level
    if preds is not None:
      preds = rep.get_inference_collapse_fn(v_seq_length)(preds)
    # Remove empty batch dimensions
    labels = tf.squeeze(labels, axis=0)
    logits = tf.squeeze(logits, axis=0)
    if preds is not None:
      preds = tf.squeeze(preds, axis=0)
    # Export probs for two stage model
    ids = [video_id] * v_seq_length
    if FLAGS.predict_mode == "probs":
      logging.info("Saving labels and probs")
      probs = tf.nn.softmax(logits, axis=-1)
      save_array = np.column_stack((ids, labels.numpy().tolist(),
        probs.numpy().tolist()))
      np.savetxt(export_csv, save_array, delimiter=",", fmt='%s')
      continue
    # Update metrics for single stage model
    metrics.update(labels, preds)
    # Save for single stage model
    logging.info("Writing {0} examples to {1}.csv...".format(len(ids), video_id))
    save_array = np.column_stack((ids, labels.numpy().tolist(),
      logits.numpy().tolist(), preds.numpy().tolist()))
    np.savetxt(export_csv, save_array, delimiter=",", fmt='%s')
  if FLAGS.predict_mode == "probs":
    # Finish
    exit()
  # Print metrics
  metrics.finish()

class Adam(tf.keras.optimizers.Adam):
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

class LossScaleOptimizer(tf.keras.mixed_precision.experimental.LossScaleOptimizer):
  def _decayed_lr(self, var_dtype):
    """Get learning rate based on epochs."""
    return self._optimizer._decayed_lr(var_dtype)
  @property
  def epochs(self):
    """Variable. The number of epochs."""
    return self._optimizer.epochs
  def finish_epoch(self):
    """Increment epoch count"""
    return self._optimizer.finish_epoch()

class ConstantLR(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Constant learning rate wrapped in LearningRateSchedule"""
  def __init__(self, learning_rate, name=None):
    super(ConstantLR, self).__init__()
    self.learning_rate = learning_rate
    self.name = name
  def __call__(self, step):
    with tf.name_scope(self.name or "Constant"):
      return self.learning_rate

def main(arg=None):
  if FLAGS.mode == 'train_and_evaluate':
    train_and_evaluate()
  elif FLAGS.mode == 'predict':
    predict()

# Run
if __name__ == "__main__":
  app.run(main=main)
