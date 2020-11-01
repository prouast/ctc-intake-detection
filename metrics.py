"""Get F1 metric based on precision and recall metrics, which in turn are based
  on true positive, true negative, and false positive."""

import tensorflow as tf
from absl import logging

@tf.function
def evaluate_interval_detection(labels, predictions, event_val, def_val, seq_length, other_vals=[]):
  """Evaluate interval detection for sequences by calculating
    tp, fp, and fn.

  Extends the metric outlined by Kyritsis et al. (2019) in
    Modeling wrist micromovements to measure in-meal eating behavior from
    inertial sensor data
    https://ieeexplore.ieee.org/abstract/document/8606156/
    by introducing additional possible events.

  Args:
    labels: The ground truth [batch_size, seq_length], encoding relevant
      sequences using the vals given in parameters.
    predictions: The predictions [batch_size, seq_length], encoding relevant
      sequences using the vals given in parameters.
    event_val: The value for true events.
    def_val: The default value for non-events.
    other_vals: List or 1-D tensor of vals for other events.
    seq_length: The sequence length.

  Returns:
    tp: True positives (number of true sequences of event_vals predicted
      with at least one predicting event_val) - scalar
    fp_1: False positives type 1 (number of excess predicting event_vals
      matching a true sequence of event_val in excess) - scalar
    fp_2: False positives type 2 (number of predicting event_vals matching
      def_val instead of event_val) - scalar
    fp_3: False positives type 3 (number of predicting event_vals matching
      other_vals instead of event_val) - 1D tensor with value for each
      element in other_vals
    fn: False negatives (number of true sequences of event_vals not matched
      by at least one predicting event_val)
  """
  def sequence_masks(labels, event_val, def_val, batch_size, seq_length):
    """Generate masks [labels, max_seq_count, seq_length] for all event sequences in the labels"""

    # Mask non-event elements as False and event elements as True
    event_mask = tf.equal(labels, event_val)

    # Mask elements that are not equal to previous elements
    diff_mask = tf.not_equal(event_mask[:, 1:], event_mask[:, :-1])
    prev_mask = tf.concat([tf.ones_like(labels[:, :1], tf.bool), diff_mask], axis=1)
    next_mask = tf.concat([diff_mask, tf.ones_like(labels[:, :1], tf.bool)], axis=1)

    # Test if there are no sequences
    empty = tf.equal(tf.reduce_sum(tf.cast(event_mask, tf.int32)), 0)

    # Mask sequence starts and ends
    seq_start_mask = tf.logical_and(prev_mask, event_mask)
    seq_end_mask = tf.logical_and(next_mask, event_mask)

    # Scatter seq_val
    seq_count_per_batch = tf.reduce_sum(tf.cast(seq_start_mask, tf.int32), axis=[1])
    max_seq_count = tf.reduce_max(seq_count_per_batch)
    seq_val_idx_mask = tf.reshape(tf.sequence_mask(seq_count_per_batch, maxlen=max_seq_count), [-1])
    seq_val_idx = tf.boolean_mask(tf.range(tf.size(seq_val_idx_mask)), seq_val_idx_mask)
    seq_vals = tf.boolean_mask(labels, seq_start_mask)
    seq_val = tf.scatter_nd(
      indices=tf.expand_dims(seq_val_idx, axis=1),
      updates=seq_vals,
      shape=tf.shape(seq_val_idx_mask))
    seq_val = tf.reshape(seq_val, [batch_size, max_seq_count])

    # Set elements of seq_val that are not event_val to def_val
    seq_val = tf.where(
      tf.not_equal(seq_val, tf.fill(tf.shape(seq_val), event_val)),
      x=tf.fill(tf.shape(seq_val), def_val), y=seq_val)

    # Scatter seq_start
    seq_start_idx = tf.where(seq_start_mask)[:,1]
    seq_start = tf.scatter_nd(
      indices=tf.expand_dims(seq_val_idx, axis=1),
      updates=seq_start_idx,
      shape=tf.shape(seq_val_idx_mask))
    seq_start = tf.reshape(seq_start, [batch_size, max_seq_count])

    # Scatter seq_end
    seq_end_idx = tf.where(seq_end_mask)[:,1]
    seq_end = tf.scatter_nd(
      indices=tf.expand_dims(seq_val_idx, axis=1),
      updates=seq_end_idx,
      shape=tf.shape(seq_val_idx_mask))
    seq_end = tf.reshape(seq_end, [batch_size, max_seq_count])

    def batch_seq_masks(starts, ends, length, vals, def_val):
      """Return seq masks for one batch"""
      def seq_mask(start, end, length, val, def_val):
        """Return one seq mask"""
        return tf.concat([
          tf.fill([start], def_val),
          tf.fill([end-start+1], val),
          tf.fill([length-end-1], def_val)], axis=0)
      return tf.map_fn(
        fn=lambda x: seq_mask(x[0], x[1], length, x[2], def_val),
        elems=(starts, ends, vals),
        dtype=tf.int32)

    seq_masks = tf.cond(empty,
      lambda: tf.fill([batch_size, 0, seq_length], def_val),
      lambda: tf.map_fn(
        fn=lambda x: batch_seq_masks(x[0], x[1], seq_length, x[2], def_val),
        elems=(seq_start, seq_end, seq_val),
        dtype=tf.int32))

    return seq_masks, max_seq_count

  labels = tf.cast(labels, dtype=tf.int32)
  predictions = tf.cast(predictions, dtype=tf.int32)
  def_val = tf.cast(def_val, dtype=tf.int32)
  event_val = tf.cast(event_val, dtype=tf.int32)

  # Dimensions
  batch_size = labels.get_shape()[0]

  # Compute whether labels are empty (no event_val sequences)
  event_mask = tf.equal(labels, event_val)
  empty = tf.equal(tf.reduce_sum(tf.cast(event_mask, tf.int32)), 0)

  # Derive positive ground truth mask; reshape to [n_gt_seq, seq_length]
  pos_mask, max_seq_count = sequence_masks(labels, event_val=event_val,
    def_val=def_val, batch_size=batch_size, seq_length=seq_length)
  pos_mask = tf.reshape(pos_mask, [-1, seq_length])

  # Mask of default events
  def_mask = tf.equal(labels, def_val)

  # Masks for other events
  other_masks = tf.map_fn(fn=lambda x: tf.equal(labels, x),
    elems=tf.convert_to_tensor(other_vals, dtype=tf.int32), dtype=tf.bool)

  # Retain only event_val in predictions
  predictions = tf.where(
    tf.not_equal(predictions, tf.fill(tf.shape(predictions), event_val)),
    x=tf.fill(tf.shape(predictions), def_val), y=predictions)

  # Stack predictions accordingly
  pred_stacked = tf.reshape(tf.tile(tf.expand_dims(predictions, axis=1), [1, max_seq_count, 1]), [-1, seq_length])

  # Remove empty masks and according preds
  keep_mask = tf.greater(tf.reduce_sum(tf.cast(tf.not_equal(pos_mask, def_val), tf.int32), axis=1), 0)
  pos_mask = tf.cond(empty,
    lambda: pos_mask,
    lambda: tf.boolean_mask(pos_mask, keep_mask))
  pred_stacked = tf.cond(empty,
    lambda: pred_stacked,
    lambda: tf.boolean_mask(pred_stacked, keep_mask))

  # Calculate number predictions per pos sequence
  # Reduce predictions to elements in pos_mask that equal event_val, then count them
  pred_sums = tf.map_fn(
    fn=lambda x: tf.reduce_sum(tf.cast(tf.equal(tf.boolean_mask(x[0], tf.equal(x[1], event_val)), event_val), tf.int32)),
    elems=(pred_stacked, pos_mask), dtype=tf.int32)

  # Calculate true positive, false positive and false negative count
  tp = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 1, lambda: 0), pred_sums))
  fn = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 0, lambda: 1), pred_sums))
  fp_1 = tf.cond(empty,
    lambda: 0,
    lambda: tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 1, lambda: count-1, lambda: 0), pred_sums)))

  # False positives of type 2 are any detections on default events
  fp_2 = tf.reduce_sum(tf.cast(tf.equal(tf.boolean_mask(predictions, def_mask), event_val), tf.int32))

  # False positives of type 3 are any detections on other events
  fp_3 = tf.map_fn(
    fn=lambda x: tf.reduce_sum(tf.cast(tf.equal(tf.boolean_mask(predictions, x), event_val), tf.int32)),
    elems=other_masks, dtype=tf.int32)

  tp = tf.cast(tp, tf.float32)
  fp_1 = tf.cast(fp_1, tf.float32)
  fp_2 = tf.cast(fp_2, tf.float32)
  fp_3 = tf.cast(fp_3, tf.float32)
  fn = tf.cast(fn, tf.float32)

  return tp, fp_1, fp_2, fp_3, fn

class TP_FP1_FP2_FP3_FN(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, seq_length, other_vals=[], log=False, name=None, dtype=None):
    super(TP_FP1_FP2_FP3_FN, self).__init__(name=name, dtype=dtype)
    self.seq_length = seq_length
    self.event_val = event_val
    self.def_val = def_val
    self.other_vals = other_vals
    self.log = log
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp_1 = self.add_weight('total_fp_1',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp_2 = self.add_weight('total_fp_2',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp_3 = self.add_weight('total_fp_3',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fn = self.add_weight('total_fn',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    seq_length = tf.shape(y_true, out_type=tf.int64)[1] if self.seq_length is None else self.seq_length
    tp, fp_1, fp_2, fp_3, fn = evaluate_interval_detection(
      labels=y_true, predictions=y_pred, event_val=self.event_val,
      def_val=self.def_val, other_vals=self.other_vals,
      seq_length=seq_length)
    self.total_tp.assign_add(tp)
    self.total_fp_1.assign_add(fp_1)
    self.total_fp_2.assign_add(fp_2)
    self.total_fp_3.assign_add(tf.reduce_sum(fp_3))
    self.total_fn.assign_add(fn)
    if self.log:
      logging.info("TP: {0}, FP1: {1}, FP2: {2}, FP3: {3}, FN: {4}".format(
        tp, fp_1, fp_2, tf.reduce_sum(fp_3), fn))

  def result(self):
    return self.total_tp, self.total_fp_1, self.total_fp_2, self.total_fp_3, self.total_fn

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fp_1.assign(0)
    self.total_fp_2.assign(0)
    self.total_fp_3.assign(0)
    self.total_fn.assign(0)

class EventPrecision(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, seq_length, other_vals=[], log=False, name=None, dtype=None):
    super(EventPrecision, self).__init__(name=name, dtype=dtype)
    self.seq_length = seq_length
    self.event_val = event_val
    self.def_val = def_val
    self.other_vals = other_vals
    self.log = log
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp = self.add_weight('total_fp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    seq_length = tf.shape(y_true, out_type=tf.int64)[1] if self.seq_length is None else self.seq_length
    tp, fp_1, fp_2, fp_3, _ = evaluate_interval_detection(
      labels=y_true, predictions=y_pred, event_val=self.event_val,
      def_val=self.def_val, other_vals=self.other_vals,
      seq_length=seq_length)
    self.total_tp.assign_add(tp)
    self.total_fp.assign_add(fp_1)
    self.total_fp.assign_add(fp_2)
    self.total_fp.assign_add(tf.reduce_sum(fp_3))
    if self.log:
      logging.info("Precision: {0}".format(
        tf.math.divide_no_nan(tp, tp+fp_1+fp_2+tf.reduce_sum(fp_3))))

  def result(self):
    return tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fp)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fp.assign(0)

class EventRecall(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, seq_length, other_vals=[], log=False, name=None, dtype=None):
    super(EventRecall, self).__init__(name=name, dtype=dtype)
    self.seq_length = seq_length
    self.event_val = event_val
    self.def_val = def_val
    self.other_vals = other_vals
    self.log = log
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fn = self.add_weight('total_fn',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    seq_length = tf.shape(y_true, out_type=tf.int64)[1] if self.seq_length is None else self.seq_length
    tp, _, _, _, fn = evaluate_interval_detection(
      labels=y_true, predictions=y_pred, event_val=self.event_val,
      def_val=self.def_val, other_vals=self.other_vals,
      seq_length=seq_length)
    self.total_tp.assign_add(tp)
    self.total_fn.assign_add(fn)
    if self.log:
      logging.info("Recall: {0}".format(
        tf.math.divide_no_nan(tp, tp+fn)))

  def result(self):
    return tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fn)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fn.assign(0)

class EventF1(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, seq_length, other_vals=[], log=False, name=None, dtype=None):
    super(EventF1, self).__init__(name=name, dtype=dtype)
    self.seq_length = seq_length
    self.event_val = event_val
    self.def_val = def_val
    self.other_vals = other_vals
    self.log = log
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fn = self.add_weight('total_fn',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp = self.add_weight('total_fp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    seq_length = tf.shape(y_true, out_type=tf.int64)[1] if self.seq_length is None else self.seq_length
    tp, fp_1, fp_2, fp_3, fn = evaluate_interval_detection(
      labels=y_true, predictions=y_pred, event_val=self.event_val,
      def_val=self.def_val, other_vals=self.other_vals,
      seq_length=seq_length)
    self.total_tp.assign_add(tp)
    self.total_fp.assign_add(fp_1)
    self.total_fp.assign_add(fp_2)
    self.total_fp.assign_add(tf.reduce_sum(fp_3))
    self.total_fn.assign_add(fn)
    if self.log:
      pre = tf.math.divide_no_nan(tp, tp+fp_1+fp_2+tf.reduce_sum(fp_3))
      rec = tf.math.divide_no_nan(tp, tp+fn)
      logging.info("F1: {0}".format(tf.math.divide_no_nan(
        2*pre*rec, pre+rec)))

  def result(self):
    pre = tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fp)
    rec = tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fn)
    return tf.math.divide_no_nan(
      2 * pre * rec, pre + rec)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fn.assign(0)
    self.total_fp.assign(0)

class FramePrecision(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, name=None, dtype=None):
    super(FramePrecision, self).__init__(name=name, dtype=dtype)
    self.event_val = event_val
    self.def_val = def_val
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp = self.add_weight('total_fp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    # Resolve frame-level predictions with argmax
    y_pred = tf.math.argmax(y_pred, axis=-1)
    # Adjust representation of precictions with DEF_VAL
    y_pred = y_pred + self.def_val
    # Calculate true positives
    tp = tf.reduce_sum(tf.where(
      tf.logical_and(tf.equal(y_pred, self.event_val),
                     tf.equal(y_true, self.event_val)),
      1, 0))
    # Calculate false positives
    fp = tf.reduce_sum(tf.where(
      tf.logical_and(tf.equal(y_pred, self.event_val),
                     tf.not_equal(y_true, self.event_val)),
      1, 0))
    # Update state
    self.total_tp.assign_add(tf.cast(tp, tf.float32))
    self.total_fp.assign_add(tf.cast(fp, tf.float32))

  def result(self):
    return tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fp)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fp.assign(0)

class FrameRecall(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, name=None, dtype=None):
    super(FrameRecall, self).__init__(name=name, dtype=dtype)
    self.event_val = event_val
    self.def_val = def_val
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fn = self.add_weight('total_fn',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    # Resolve frame-level predictions with argmax
    y_pred = tf.math.argmax(y_pred, axis=-1)
    # Adjust representation of precictions with DEF_VAL
    y_pred = y_pred + self.def_val
    # Calculate true positives
    tp = tf.reduce_sum(tf.where(
      tf.logical_and(tf.equal(y_pred, self.event_val),
                     tf.equal(y_true, self.event_val)),
      1, 0))
    # Calculate false negatives
    fn = tf.reduce_sum(tf.where(
      tf.logical_and(tf.not_equal(y_pred, self.event_val),
                     tf.equal(y_true, self.event_val)),
      1, 0))
    # Update state
    self.total_tp.assign_add(tf.cast(tp, tf.float32))
    self.total_fn.assign_add(tf.cast(fn, tf.float32))

  def result(self):
    return tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fn)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fn.assign(0)

class FrameF1(tf.keras.metrics.Metric):
  def __init__(self, event_val, def_val, name=None, dtype=None):
    super(FrameF1, self).__init__(name=name, dtype=dtype)
    self.event_val = event_val
    self.def_val = def_val
    self.total_tp = self.add_weight('total_tp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fn = self.add_weight('total_fn',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
    self.total_fp = self.add_weight('total_fp',
      shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

  def update_state(self, y_true, y_pred):
    # Resolve frame-level predictions with argmax
    y_pred = tf.math.argmax(y_pred, axis=-1)
    # Adjust representation of precictions with DEF_VAL
    y_pred = y_pred + self.def_val
    # Calculate true positives
    tp = tf.reduce_sum(tf.where(
      tf.logical_and(tf.equal(y_pred, self.event_val),
                     tf.equal(y_true, self.event_val)),
      1, 0))
    # Calculate false positives
    fp = tf.reduce_sum(tf.where(
      tf.logical_and(tf.equal(y_pred, self.event_val),
                     tf.not_equal(y_true, self.event_val)),
      1, 0))
    # Calculate false negatives
    fn = tf.reduce_sum(tf.where(
      tf.logical_and(tf.not_equal(y_pred, self.event_val),
                     tf.equal(y_true, self.event_val)),
      1, 0))
    # Update state
    self.total_tp.assign_add(tf.cast(tp, tf.float32))
    self.total_fp.assign_add(tf.cast(fp, tf.float32))
    self.total_fn.assign_add(tf.cast(fn, tf.float32))

  def result(self):
    pre = tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fp)
    rec = tf.math.divide_no_nan(
      self.total_tp,
      self.total_tp + self.total_fn)
    return tf.math.divide_no_nan(
      2 * pre * rec, pre + rec)

  def reset_states(self):
    self.total_tp.assign(0)
    self.total_fn.assign(0)
    self.total_fp.assign(0)

class TrainMetrics():
  def __instantiate_event_metrics(self):
    self.metrics = {
      'mean_precision': tf.keras.metrics.Mean(),
      'mean_recall': tf.keras.metrics.Mean(),
      'mean_f1': tf.keras.metrics.Mean()}
    for i in self.event_classes:
      other_vals = [j for j in self.event_classes if j != i]
      self.metrics['class_{}_precision'.format(i)] = EventPrecision(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)
      self.metrics['class_{}_recall'.format(i)] = EventRecall(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)
      self.metrics['class_{}_f1'.format(i)] = EventF1(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)

  def __instantiate_frame_metrics(self):
    self.metrics = {
      'mean_precision': tf.keras.metrics.Mean(),
      'mean_recall': tf.keras.metrics.Mean(),
      'mean_f1': tf.keras.metrics.Mean()}
    for i in self.all_classes:
      self.metrics['class_{}_precision'.format(i)] = FramePrecision(
        event_val=i, def_val=self.def_val)
      self.metrics['class_{}_recall'.format(i)] = FrameRecall(
        event_val=i, def_val=self.def_val)
      self.metrics['class_{}_f1'.format(i)] = FrameF1(
        event_val=i, def_val=self.def_val)

  def __init__(self, representation, writer):
    self.def_val = representation.get_def_val()
    self.event_classes = representation.get_event_classes_range()
    self.all_classes = representation.get_all_classes_range()
    self.loss_mode = representation.get_loss_mode()
    self.seq_length = representation.get_seq_length()
    self.writer = writer
    # Instantiate the metrics
    if self.loss_mode == "ctc":
      self.__instantiate_event_metrics()
    else:
      self.__instantiate_frame_metrics()

  def update(self, labels, logits, predictions_u):
    if self.loss_mode == "ctc":
      for i in self.event_classes:
        self.metrics['class_{}_precision'.format(i)](labels, predictions_u)
        self.metrics['class_{}_recall'.format(i)](labels, predictions_u)
        self.metrics['class_{}_f1'.format(i)](labels, predictions_u)
        self.metrics['mean_precision'](self.metrics['class_{}_precision'.format(i)].result())
        self.metrics['mean_recall'](self.metrics['class_{}_recall'.format(i)].result())
        self.metrics['mean_f1'](self.metrics['class_{}_f1'.format(i)].result())
    else:
      # Collapse seq_length dim if 1
      labels = tf.squeeze(labels, axis=1)
      logits = tf.squeeze(logits, axis=1)
      for i in self.all_classes:
        self.metrics['class_{}_precision'.format(i)](labels, logits)
        self.metrics['class_{}_recall'.format(i)](labels, logits)
        self.metrics['class_{}_f1'.format(i)](labels, logits)
        self.metrics['mean_precision'](self.metrics['class_{}_precision'.format(i)].result())
        self.metrics['mean_recall'](self.metrics['class_{}_recall'.format(i)].result())
        self.metrics['mean_f1'](self.metrics['class_{}_f1'.format(i)].result())

  def log(self, global_step):
    classes = self.event_classes if self.loss_mode == "ctc" else self.all_classes
    logging.info('Mean training precision (this step): {}'.format(float(self.metrics['mean_precision'].result())))
    logging.info('Mean training recall (this step): {}'.format(float(self.metrics['mean_recall'].result())))
    logging.info('Mean training f1 (this step): {}'.format(float(self.metrics['mean_f1'].result())))
    # TensorBoard
    with self.writer.as_default():
      tf.summary.scalar('metrics/mean_precision', data=self.metrics['mean_precision'].result(), step=global_step)
      tf.summary.scalar('metrics/mean_recall', data=self.metrics['mean_recall'].result(), step=global_step)
      tf.summary.scalar('metrics/mean_f1', data=self.metrics['mean_f1'].result(), step=global_step)
    # Reset metrics
    self.metrics['mean_precision'].reset_states()
    self.metrics['mean_recall'].reset_states()
    self.metrics['mean_f1'].reset_states()
    # For each class
    for i in classes:
      # Get metrics
      pre = self.metrics['class_{}_precision'.format(i)].result()
      rec = self.metrics['class_{}_recall'.format(i)].result()
      f1 = self.metrics['class_{}_f1'.format(i)].result()
      # Console
      logging.info('Class {} training precision (this step): {}'.format(i, float(pre)))
      logging.info('Class {} training recall (this step): {}'.format(i, float(rec)))
      logging.info('Class {} training f1 (this step): {}'.format(i, float(f1)))
      # TensorBoard
      with self.writer.as_default():
        tf.summary.scalar('metrics/class_{}_precision'.format(i), data=pre, step=global_step)
        tf.summary.scalar('metrics/class_{}_recall'.format(i), data=rec, step=global_step)
        tf.summary.scalar('metrics/class_{}_f1'.format(i), data=f1, step=global_step)
      # Reset metrics
      self.metrics['class_{}_precision'.format(i)].reset_states()
      self.metrics['class_{}_recall'.format(i)].reset_states()
      self.metrics['class_{}_f1'.format(i)].reset_states()

class EvalMetrics():
  def __instantiate_event_metrics(self):
    self.metrics = {
      'mean_precision': tf.keras.metrics.Mean(),
      'mean_recall': tf.keras.metrics.Mean(),
      'mean_f1': tf.keras.metrics.Mean()}
    for i in self.event_classes:
      other_vals = [j for j in self.event_classes if j != i]
      self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)] = TP_FP1_FP2_FP3_FN(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)
      self.metrics['class_{}_precision'.format(i)] = EventPrecision(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)
      self.metrics['class_{}_recall'.format(i)] = EventRecall(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)
      self.metrics['class_{}_f1'.format(i)] = EventF1(
        event_val=i, def_val=self.def_val, seq_length=self.seq_length,
        other_vals=other_vals)

  def __instantiate_frame_metrics(self):
    self.metrics = {
      'mean_precision': tf.keras.metrics.Mean(),
      'mean_recall': tf.keras.metrics.Mean(),
      'mean_f1': tf.keras.metrics.Mean()}
    for i in self.all_classes:
      self.metrics['class_{}_precision'.format(i)] = FramePrecision(
        event_val=i, def_val=self.def_val)
      self.metrics['class_{}_recall'.format(i)] = FrameRecall(
        event_val=i, def_val=self.def_val)
      self.metrics['class_{}_f1'.format(i)] = FrameF1(
        event_val=i, def_val=self.def_val)

  def __init__(self, representation, writer):
    self.def_val = representation.get_def_val()
    self.writer = writer
    self.event_classes = representation.get_event_classes_range()
    self.all_classes = representation.get_all_classes_range()
    self.loss_mode = representation.get_loss_mode()
    self.seq_length = representation.get_seq_length()
    # Instantiate the metrics
    if self.loss_mode == "ctc":
      self.__instantiate_event_metrics()
    else:
      self.__instantiate_frame_metrics()

  def update_i(self, labels, logits, predictions_u):
    # Collapse seq_length dim if 1
    labels = tf.squeeze(labels, axis=1)
    logits = tf.squeeze(logits, axis=1)
    if self.loss_mode == "ctc":
      for i in self.event_classes:
        self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)](labels, predictions_u)
        self.metrics['class_{}_precision'.format(i)](labels, predictions_u)
        self.metrics['class_{}_recall'.format(i)](labels, predictions_u)
        self.metrics['class_{}_f1'.format(i)](labels, predictions_u)
    else:
      for i in self.all_classes:
        self.metrics['class_{}_precision'.format(i)](labels, logits)
        self.metrics['class_{}_recall'.format(i)](labels, logits)
        self.metrics['class_{}_f1'.format(i)](labels, logits)

  def update(self):
    classes = self.event_classes if self.loss_mode == "ctc" else self.all_classes
    for i in classes:
      self.metrics['mean_precision'](self.metrics['class_{}_precision'.format(i)].result())
      self.metrics['mean_recall'](self.metrics['class_{}_recall'.format(i)].result())
      self.metrics['mean_f1'](self.metrics['class_{}_f1'.format(i)].result())
    return self.metrics['mean_f1'].result()

  def log(self, global_step):
    classes = self.event_classes if self.loss_mode == "ctc" else self.all_classes
    logging.info('Mean eval precision: {}'.format(float(self.metrics['mean_precision'].result())))
    logging.info('Mean eval recall: {}'.format(float(self.metrics['mean_recall'].result())))
    logging.info('Mean eval f1: {}'.format(float(self.metrics['mean_f1'].result())))
    # TensorBoard
    with self.writer.as_default():
      tf.summary.scalar('metrics/mean_precision', data=self.metrics['mean_precision'].result(), step=global_step)
      tf.summary.scalar('metrics/mean_recall', data=self.metrics['mean_recall'].result(), step=global_step)
      tf.summary.scalar('metrics/mean_f1', data=self.metrics['mean_f1'].result(), step=global_step)
    # Reset metrics
    self.metrics['mean_precision'].reset_states()
    self.metrics['mean_recall'].reset_states()
    self.metrics['mean_f1'].reset_states()
    # For each class
    for i in classes:
      # Get metrics
      if self.loss_mode == "ctc":
        tp, fp1, fp2, fp3, fn = self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)].result()
      pre = self.metrics['class_{}_precision'.format(i)].result()
      rec = self.metrics['class_{}_recall'.format(i)].result()
      f1 = self.metrics['class_{}_f1'.format(i)].result()
      # Console
      if self.loss_mode == "ctc":
        logging.info('Class {} eval tp: {}, fp1: {}, fp2: {}, fp3: {}, fn: {}'.format(
          i, int(tp), int(fp1), int(fp2), int(fp3), int(fn)))
      logging.info('Class {} eval precision: {}'.format(i, float(pre)))
      logging.info('Class {} eval recall: {}'.format(i, float(rec)))
      logging.info('Class {} eval f1: {}'.format(i, float(f1)))
      # TensorBoard
      with self.writer.as_default():
        if self.loss_mode == "ctc":
          tf.summary.scalar('metrics/class_{}_tp'.format(i), data=tp, step=global_step)
          tf.summary.scalar('metrics/class_{}_fp_1'.format(i), data=fp1, step=global_step)
          tf.summary.scalar('metrics/class_{}_fp_2'.format(i), data=fp2, step=global_step)
          tf.summary.scalar('metrics/class_{}_fp_3'.format(i), data=fp3, step=global_step)
          tf.summary.scalar('metrics/class_{}_fn'.format(i), data=fn, step=global_step)
        tf.summary.scalar('metrics/class_{}_precision'.format(i), data=pre, step=global_step)
        tf.summary.scalar('metrics/class_{}_recall'.format(i), data=rec, step=global_step)
        tf.summary.scalar('metrics/class_{}_f1'.format(i), data=f1, step=global_step)
      # Reset eval metric states after evaluation
      if self.loss_mode == "ctc":
        self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)].reset_states()
      self.metrics['class_{}_precision'.format(i)].reset_states()
      self.metrics['class_{}_recall'.format(i)].reset_states()
      self.metrics['class_{}_f1'.format(i)].reset_states()

class PredMetrics():
  def __instantiate_event_metrics(self):
    self.metrics = {}
    for i in self.event_classes:
      other_vals = [j for j in self.event_classes if j != i]
      self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)] = TP_FP1_FP2_FP3_FN(
        event_val=i, def_val=self.def_val, seq_length=None,
        other_vals=other_vals, log=True)
      self.metrics['class_{}_precision'.format(i)] = EventPrecision(
        event_val=i, def_val=self.def_val, seq_length=None,
        other_vals=other_vals, log=True)
      self.metrics['class_{}_recall'.format(i)] = EventRecall(
        event_val=i, def_val=self.def_val, seq_length=None,
        other_vals=other_vals, log=True)
      self.metrics['class_{}_f1'.format(i)] = EventF1(
        event_val=i, def_val=self.def_val, seq_length=None,
        other_vals=other_vals, log=True)

  def __init__(self, representation):
    self.def_val = representation.get_def_val()
    self.event_classes = representation.get_event_classes_range()
    self.num_event_classes = representation.get_num_event_classes()
    # Instantiate the metrics
    self.__instantiate_event_metrics()

  def update(self, labels, preds):
    labels = tf.expand_dims(labels, 0)
    preds = tf.expand_dims(preds, 0)
    for i in self.event_classes:
      logging.info("---------------------- Class {} --------------------".format(i))
      # Calculate metrics
      tp, fp_1, fp_2, fp_3, fn = self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)](labels, preds)
      self.metrics['class_{}_precision'.format(i)](labels, preds)
      self.metrics['class_{}_recall'.format(i)](labels, preds)
      f1 = self.metrics['class_{}_f1'.format(i)](labels, preds)
    logging.info("===================================================")

  def finish(self):
    logging.info("===================== Finished ====================")
    m_pre = 0; m_rec = 0; m_f1 = 0;
    for i in self.event_classes:
      logging.info("---------------------- Class {} --------------------".format(i))
      tp, fp1, fp2, fp3, fn = self.metrics['class_{}_tp_fp1_fp2_fp3_fn'.format(i)].result()
      pre = self.metrics['class_{}_precision'.format(i)].result()
      rec = self.metrics['class_{}_recall'.format(i)].result()
      f1 = self.metrics['class_{}_f1'.format(i)].result()
      logging.info("TP: {0}, FP1: {1}, FP2: {2}, FP3: {3}, FN: {4}".format(
        tp, fp1, fp2, fp3, fn))
      logging.info("Precision: {0}, Recall: {1}".format(pre, rec))
      logging.info("F1: {0}".format(f1))
      m_pre += pre; m_rec += rec; m_f1 += f1
    logging.info("===================================================")
    logging.info("mPrecision: {0}, mRecall: {1}".format(
      m_pre/self.num_event_classes, m_rec/self.num_event_classes))
    logging.info("mF1: {0}".format(m_f1/self.num_event_classes))
