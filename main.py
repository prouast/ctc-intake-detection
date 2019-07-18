import os
import math
import numpy as np
import tensorflow as tf
import best_checkpoint_exporter
from tensorflow.python.platform import gfile
import itertools
import oreba_cnn_lstm
import lstm


ORIGINAL_SIZE = 140
FRAME_SIZE = 128
NUM_CHANNELS = 3
SEQ_LENGTH = 16
NUM_SHARDS = 10
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    name='batch_size', default=32, help='Batch size used for training.')
tf.app.flags.DEFINE_string(
    name='eval_dir', default='data/raw/eval', help='Directory for eval data.')
tf.app.flags.DEFINE_enum(
    name='input_mode', default="raw", enum_values=["fc7", "raw"],
    help='What does the input data look like.')
tf.app.flags.DEFINE_enum(
    name='mode', default="train_and_evaluate", enum_values=["train_and_evaluate", "predict"],
    help='What mode should tensorflow be started in')
tf.app.flags.DEFINE_string(
    name='model', default='oreba_cnn_lstm',
    help='Select the model: {oreba_cnn_lstm, lstm}')
tf.app.flags.DEFINE_string(
    name='model_dir', default='run',
    help='Output directory for model and training stats.')
tf.app.flags.DEFINE_integer(
    name='num_features', default=2048, help='Number of fc7 features as input.')
tf.app.flags.DEFINE_integer(
    name='num_seq', default=396960, help='Number of training sequences.')
tf.app.flags.DEFINE_integer(
    name='seq_length', default=16,
    help='Number of sequence elements.')
tf.app.flags.DEFINE_integer(
    name='seq_shift', default=1,
    help='Number of sequence elements.')
tf.app.flags.DEFINE_string(
    name='train_dir', default='data/raw/train', help='Directory for training data.')
tf.app.flags.DEFINE_float(
    name='train_epochs', default=60, help='Number of training epochs.')


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
        decay_rate=0.94,
        dropout=0.5,
        gradient_clipping_norm=10.0,
        num_classes=2,
        seq_length=FLAGS.seq_length,
        steps_per_epoch=steps_per_epoch)

    # Run config
    run_config = tf.estimator.RunConfig(
        model_dir="run",
        save_summary_steps=10,
        save_checkpoints_steps=50)

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
    if FLAGS.model == "oreba_cnn_lstm":
        model = oreba_cnn_lstm.Model(params)
    elif FLAGS.model == "lstm":
        model = lstm.Model(params)

    logits = model(features, is_training)

    # Decode logits into predictions
    predictions, _ = greedy_decode_with_indices(logits, params.num_classes, params.seq_length)

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

    def dense_to_sparse(input, eos_token=0):
        idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
        values = tf.gather_nd(input, idx)
        shape = tf.shape(input, out_type=tf.int64)
        sparse = tf.SparseTensor(idx, values, shape)
        return sparse

    # Calculate ctc loss from SparseTensor without collapsing labels
    # Works with preprocess_collapse_repeated=False, ctc_merge_repeated=False (implication that labelling could be simplified!)
    # Also works with preprocess_collapse_repeated=True, ctc_merge_repeated=False
    seq_length = tf.fill([params.batch_size], params.seq_length)
    loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(labels, eos_token=-1),
        inputs=logits,
        sequence_length=seq_length,
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
    pre, rec, pre_op, rec_op = pre_rec(labels, predictions, FLAGS.seq_length, evaluate_interval_detection)
    f1, f1_op = f1_metric(labels, predictions, FLAGS.seq_length, evaluate_interval_detection)

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


def collapse_sequences(labels, seq_length, def_val=1, pad_val=0, replace_with_idle=True, pos='middle'):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 1) and
            sequences of interest (e.g., [1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 1]).
        def_val: The value which denotes the default value
        replace_with_idle: If true, collapsed values are replaced with the
            idle val. If false, collapsed values are removed (Tensor is padded
            with 0).
        pos: The position relative to the original sequence to keep the
            remaining non-collapsed value.
    """
    # Get general dimensions
    batch_size = labels.get_shape()[0]
    maxlen = seq_length

    # Find differences in labels
    diff_mask = tf.not_equal(labels[:, 1:], labels[:, :-1])

    # Mask labels that don't equal previous/next label.
    prev_mask = tf.concat([tf.ones_like(labels[:, :1], tf.bool), diff_mask], axis=1)
    next_mask = tf.concat([diff_mask, tf.ones_like(labels[:, :1], tf.bool)], axis=1)

    if replace_with_idle:

        # Masks for non-default vals, sequence starts and ends
        not_default_mask = tf.not_equal(labels, tf.fill(tf.shape(labels), def_val))
        seq_start_mask = tf.logical_and(prev_mask, not_default_mask)
        seq_end_mask = tf.logical_and(next_mask, not_default_mask)

        # Test if there are no sequences in labels
        empty = tf.equal(tf.reduce_sum(tf.cast(not_default_mask, tf.int32)), 0)

        # Sequence val occurrences
        seq_vals = tf.boolean_mask(labels, seq_start_mask)

        # Prepare padded seq vals
        seq_count_per_batch = tf.reduce_sum(tf.cast(seq_start_mask, tf.int32), axis=[1])
        max_seq_count = tf.reduce_max(seq_count_per_batch)
        seq_val_idx_mask = tf.reshape(tf.sequence_mask(seq_count_per_batch, maxlen=max_seq_count), [-1])
        seq_val_idx = tf.boolean_mask(tf.range(tf.size(seq_val_idx_mask)), seq_val_idx_mask)
        seq_val = tf.scatter_nd(
            indices=tf.expand_dims(seq_val_idx, axis=1),
            updates=seq_vals,
            shape=tf.shape(seq_val_idx_mask))
        seq_val = tf.reshape(seq_val, [batch_size, max_seq_count])

        # Prepare padded seq idx
        seq_se_count_per_batch = tf.reduce_sum(
            tf.cast(seq_start_mask, tf.int32) + tf.cast(seq_end_mask, tf.int32), axis=[1])
        max_seq_se_count = tf.reduce_max(seq_se_count_per_batch)
        se_idx_mask = tf.reshape(tf.sequence_mask(seq_se_count_per_batch,
            maxlen=max_seq_se_count), [-1])
        se_idx = tf.boolean_mask(tf.range(tf.size(se_idx_mask)), se_idx_mask)
        start_updates = tf.cast(tf.where(seq_start_mask)[:,1], tf.int32)
        end_updates = tf.cast(tf.where(seq_end_mask)[:,1], tf.int32)
        se_updates = tf.reshape(tf.stack([start_updates, end_updates], axis=1), [-1])
        seq_idx_se = tf.scatter_nd(
            indices=tf.expand_dims(se_idx, axis=1),
            updates=se_updates,
            shape=tf.shape(se_idx_mask))
        seq_idx_se = tf.reshape(seq_idx_se, [batch_size, max_seq_count, 2])
        seq_idx_se = tf.transpose(seq_idx_se, [0, 2, 1]) # [batch_size, start/end, num_vals]

        # For each sequence of seq_val, find the index to collapse to
        def collapse_seq_idx(start, end, pos='middle'):
            if pos == 'middle':
                return tf.floordiv(end + start, 2)
            elif pos == 'start':
                return start
            elif pos == 'end':
                return end
        seq_idx = tf.map_fn(lambda x: collapse_seq_idx(x[0], x[1]), seq_idx_se, dtype=tf.int32)

        # Scatter seq_vals and seq_idx to determine collapsed labels
        updates = tf.map_fn(
            fn=lambda x: tf.scatter_nd(indices=tf.expand_dims(x[0], axis=1),
                updates=x[1], shape=[maxlen]),
            elems=(seq_idx, seq_val),
            dtype=tf.int32)
        updates = tf.where(tf.equal(updates, 0), tf.fill(tf.shape(updates), def_val), updates)

        # Flatten and determine idx
        new_maxlen = maxlen
        flat_idx = tf.range(tf.size(labels))
        flat_updates = tf.reshape(updates, [-1])
        flat_shape = tf.shape(tf.reshape(labels, [-1]))

        seq_length = tf.fill([batch_size], maxlen)

    else:

        # Mask for all def_val in the sequence
        default_mask = tf.equal(labels, tf.fill(tf.shape(labels), def_val))

        # Combine with mask for all first sequence elements
        mask = tf.logical_or(default_mask, prev_mask)
        flat_updates = tf.boolean_mask(labels, mask, axis=0)

        # Determine new sequence lengths / max length
        new_seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        new_maxlen = tf.reduce_max(new_seq_len)

        # Mask idx
        idx_mask = tf.sequence_mask(new_seq_len, maxlen=new_maxlen)
        flat_idx_mask = tf.reshape(idx_mask, [-1])
        idx = tf.range(tf.size(idx_mask))
        flat_idx = tf.boolean_mask(idx, flat_idx_mask, axis=0)
        flat_shape = tf.shape(flat_idx_mask)

        seq_length = new_seq_len

    flat = tf.scatter_nd(
        indices=tf.expand_dims(flat_idx, axis=1),
        updates=flat_updates + 1,
        shape=flat_shape)

    flat = tf.where(tf.equal(flat, 0), tf.fill(flat_shape, pad_val + 1), flat)
    flat = flat - 1

    # Reshape back to square batch.
    new_shape = [batch_size, new_maxlen]
    result = tf.reshape(flat, new_shape)

    return result, seq_length


def greedy_decode_with_indices(inputs, num_classes, seq_length, pos='middle'):
    """Naive inference by retrieving most likely output at each time-step.

    Args:
        inputs: The prediction in form of logits. [samples, time_steps, num_classes+1]
            Contains an extra class for CTC epsilon at the last index
        num_classes: The number of classes considered in prediction.
        seq_length: Sequence length for each item in inputs.
        pos: How should predictions (excluding 0) be collapsed to 0?
    Returns:
        Tuple
        * the decoded sequence [seq_length]
        * indices of 1 predictions
    """
    def decode(input):
        cat_ids = tf.cast(tf.argmax(input, axis=1), tf.int32)
        cat_ids = tf.where(tf.equal(cat_ids, num_classes),
            tf.zeros([seq_length], tf.int32), cat_ids) # Set epsilons to 0
        row_ids = tf.range(tf.shape(input)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, cat_ids], axis=1)
        return idx[:,1]

    decoded = tf.map_fn(decode, inputs, dtype=tf.int32)
    collapsed, _ = collapse_sequences(decoded, seq_length, def_val=0, pad_val=-1,
        replace_with_idle=True, pos=pos)
    one_indices = tf.where(tf.equal(collapsed, tf.constant(1, tf.int32)))

    return collapsed, one_indices


def evaluate_interval_detection(labels, predictions, def_val, seq_length):
    """Evaluate interval detection for sequences by calculating
        tp, fp, and fn.

    Follows the metric outlined by Kyritsis et al. (2019) in
        Modeling wrist micromovements to measure in-meal eating behavior from
        inertial sensor data
        https://ieeexplore.ieee.org/abstract/document/8606156/

    Args:
        labels: The truth, where 1 means part of the sequence and 0 otherwise.
            [batch_size, seq_length]
        predictions: The predictions, where 1 means part of the sequence and 0
            otherwise. [batch_size, seq_length]
        seq_length: The sequence length.

    Returns:
        tp: True positives (number of true sequences of 1s predicted with at
                least one predicting 1)
        fn: False negatives (number of true sequences of 1s not matched by at
                least one predicting 1)
        fp: False positives (number of excess predicting 1s matching a true
                sequence of 1s in excess + number of predicting 1s not matching
                a true sequence of 1s)
    """
    def sequence_masks(labels, def_val, seq_length):
        # Get dimensions
        batch_size = labels.get_shape()[0]

        # Mask elements non-equal to previous elements
        diff_mask = tf.not_equal(labels[:, 1:], labels[:, :-1])
        prev_mask = tf.concat([tf.ones_like(labels[:, :1], tf.bool), diff_mask], axis=1)
        next_mask = tf.concat([diff_mask, tf.ones_like(labels[:, :1], tf.bool)], axis=1)

        # Mask elements that are not def_val
        not_default_mask = tf.not_equal(labels, tf.fill(tf.shape(labels), def_val))

        # Test if there are no sequences
        empty = tf.equal(tf.reduce_sum(tf.cast(not_default_mask, tf.int32)), 0)

        # Mask sequence starts and ends
        seq_start_mask = tf.logical_and(prev_mask, not_default_mask)
        seq_end_mask = tf.logical_and(next_mask, not_default_mask)

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
            def seq_mask(start, end, length, val, def_val):
                return tf.concat([
                    tf.fill([start], def_val),
                    tf.fill([end-start+1], val),
                    tf.fill([length-end-1], def_val)], axis=0)
            return tf.map_fn(
                fn=lambda x: seq_mask(x[0], x[1], length, x[2], def_val),
                elems=(starts, ends, vals),
                dtype=tf.int32)

        seq_masks = tf.cond(empty,
            lambda: tf.fill([batch_size, 1, seq_length], def_val),
            lambda: tf.map_fn(
                fn=lambda x: batch_seq_masks(x[0], x[1], seq_length, x[2], def_val),
                elems=(seq_start, seq_end, seq_val),
                dtype=tf.int32))

        return seq_masks, max_seq_count

    # Dimensions
    batch_size = labels.get_shape()[0]

    # Mask of negative ground truth
    neg_mask = tf.equal(labels, def_val)

    # Compute whether labels are empty (no sequences)
    pos_length = tf.reduce_sum(labels)
    empty = tf.cond(tf.equal(pos_length, 0), lambda: True, lambda: False)

    # Derive positive ground truth mask and stack predictions accordingly
    test = sequence_masks(labels, 0, seq_length)
    pos_mask, max_seq_count = sequence_masks(labels, 0, seq_length)
    pos_mask = tf.reshape(pos_mask, [-1, seq_length])
    pred_stacked = tf.reshape(tf.tile(tf.expand_dims(predictions, axis=1), [1, max_seq_count, 1]), [-1, seq_length])

    # Remove empty masks
    empty_mask = tf.greater(tf.reduce_sum(pos_mask, axis=1), 0)
    pos_mask = tf.boolean_mask(pos_mask, empty_mask)
    pred_stacked = tf.boolean_mask(pred_stacked, empty_mask)

    # Calculate number of predictions for pos sequences
    pred_sums = tf.map_fn(
        fn=lambda x: tf.reduce_sum(tf.boolean_mask(x[0], x[1])),
        elems=(pred_stacked, pos_mask), dtype=tf.int32)

    # Calculate true positive, false positive and false negative count
    tp = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 1, lambda: 0), pred_sums))
    fn = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 0, lambda: 1), pred_sums))
    fp = tf.cond(empty,
        lambda: 0,
        lambda: tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 1, lambda: count-1, lambda: 0), pred_sums)))
    fp += tf.reduce_sum(tf.boolean_mask(predictions, mask=neg_mask))

    tp = tf.cast(tp, tf.float32)
    fn = tf.cast(fn, tf.float32)
    fp = tf.cast(fp, tf.float32)

    return tp, fn, fp


def _aggregate_across_replicas(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across replicas."""
    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution.extended, '_outer_control_flow_context'):
            # pylint: disable=protected-access
            if distribution.extended._outer_control_flow_context is None:
                with tf.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution.extended._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution.extended._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            tf.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return tf.distribute.get_replica_context().merge_call(fn, args=args)


def _aggregate_variable(v, collections):
    f = lambda distribution, value: distribution.read_var(value)
    return _aggregate_across_replicas(collections, f, v)


def _metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.
  If running in a `DistributionStrategy` context, the variable will be
  "replica local". This means:
  *   The returned object will be a container with separate variables
      per replica of the model.
  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica.
  *   To get a metric's result value, we need to sum the variable values
      across the replicas before computing the final answer. Furthermore,
      the final answer should be computed once instead of in every
      replica. Both of these are accomplished by running the computation
      of the final result value inside
      `distribution_strategy_context.get_replica_context().merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a replica-local variable in a computation returns
      the sum across all replicas.
  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.
  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a replica-local variable container.
  """
  # Note that synchronization "ON_READ" implies trainable=False.
  return tf.Variable(
      lambda: tf.zeros(shape, dtype),
      collections=[
          tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      synchronization=tf.VariableSynchronization.ON_READ,
      aggregation=tf.VariableAggregation.SUM,
      name=name)


def tp_fn_fp(labels, predictions, seq_length, metric_fn, metrics_collections=None, updates_collections=None):
    """Metric returning true positives, false negatives and false positives for
        sequence predictions.

    Args:
        labels: The labels
        predictions: The predictions
        seq_length: The sequence length.
        metric_fn: Function calculating tp, fn, fp.
        metrics_collections:
        updates_collections:
    """
    with tf.variable_scope('tp_fn_fp', (labels, predictions)):

        total_tp = _metric_variable([], tf.float32, name='total_tp')
        total_fn = _metric_variable([], tf.float32, name='total_fn')
        total_fp = _metric_variable([], tf.float32, name='total_fp')

        tp_tensor = _aggregate_variable(total_tp, metrics_collections)
        fn_tensor = _aggregate_variable(total_fn, metrics_collections)
        fp_tensor = _aggregate_variable(total_fp, metrics_collections)

        tp, fn, fp = metric_fn(labels, predictions, 0, seq_length)

        tp_update_op = tf.assign_add(total_tp, tp)
        fn_update_op = tf.assign_add(total_fn, fn)
        fp_update_op = tf.assign_add(total_fp, fp)

        if updates_collections:
            tf.add_to_collections(updates_collections, tp_update_op)
            tf.add_to_collections(updates_collections, fn_update_op)
            tf.add_to_collections(updates_collections, fp_update_op)

    return tp_tensor, fn_tensor, fp_tensor, tp_update_op, fn_update_op, fp_update_op


def pre_rec(labels, predictions, seq_length, metric_fn, metrics_collections=None, updates_collections=None):
    """Metric returning precision and recall for sequence predictions.

    Args:
        labels: The labels
        predictions: The predictions
        seq_length: The sequence length.
        metric_fn: Function calculating the underlying tp, fn, fp for each
            batch element.
        metrics_collections:
        updates_collections:
    """
    with tf.variable_scope('pre_rec', (labels, predictions)):

        tp, fn, fp, tp_update_op, fn_update_op, fp_update_op = tp_fn_fp(
            labels, predictions, seq_length, metric_fn,
            metrics_collections=None, updates_collections=None)

        def compute_precision(tp, fp, name):
            return tf.where(
                tf.greater(tp + fp, 0), tf.divide(tp, tp + fp), 0, name)
        def compute_recall(tp, fn, name):
            return tf.where(
                tf.greater(tp + fn, 0), tf.divide(tp, tp + fn), 0, name)

        pre = _aggregate_across_replicas(
            metrics_collections, lambda _, tp, fp: compute_precision(tp, fp, 'value'), tp, fp)
        rec = _aggregate_across_replicas(
            metrics_collections, lambda _, tp, fn: compute_recall(tp, fn, 'value'), tp, fn)

        pre_update_op = compute_precision(tp_update_op, fp_update_op, 'update_op')
        rec_update_op = compute_recall(tp_update_op, fn_update_op, 'update_op')

        if updates_collections:
            tf.add_to_collections(updates_collections, pre_update_op)
            tf.add_to_collections(updates_collections, rec_update_op)

        return pre, rec, pre_update_op, rec_update_op


def f1_metric(labels, predictions, seq_length, metric_fn, metrics_collections=None, updates_collections=None):
    """Metric returning f1 for sequence predictions.

    Args:
        labels: The labels
        predictions: The predictions
        seq_length: The sequence length.
        metric_fn: Function calculating the underlying tp, fn, fp for each
            batch element.
        metrics_collections:
        updates_collections:
    """
    with tf.variable_scope('f1', (labels, predictions)):

        pre, rec, pre_update_op, rec_update_op = pre_rec(
            labels, predictions, seq_length, metric_fn,
            metrics_collections=None, updates_collections=None)

        def compute_f1(pre, rec, name):
            return tf.where(
                tf.greater(pre + rec, 0), tf.divide(2 * pre * rec, pre + rec), 0, name)

        f1 = _aggregate_across_replicas(
            metrics_collections, lambda _, pre, rec: compute_f1(pre, rec, 'value'), pre, rec)

        f1_update_op = compute_f1(pre_update_op, rec_update_op, 'update_op')

        if updates_collections:
            tf.add_to_collections(updates_collections, f1_update_op)

        return f1, f1_update_op


def input_fn(is_training, data_dir):
    """Input pipeline"""
    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecords"))
    if not filenames:
        raise RuntimeError('No files found.')
    tf.logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_SHARDS)
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=_get_input_parser(), num_parallel_calls=2)
            .apply(_get_sequence_batch_fn(is_training))
            .map(map_func=_get_transformation_parser(is_training),
                num_parallel_calls=2),
        cycle_length=NUM_SHARDS)
    if is_training:
        dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    return dataset


def _get_input_parser():

    def input_parser_fc7(serialized_example):
        features = tf.parse_single_example(
            serialized_example, {
                'example/label': tf.FixedLenFeature([], dtype=tf.int64),
                'example/fc7': tf.FixedLenFeature([FLAGS.num_features], dtype=tf.float32)
        })
        label = tf.cast(features['example/label'], tf.int32)
        fc7 = features['example/fc7']
        return fc7, label

    def input_parser_raw(serialized_example):
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

    if FLAGS.input_mode == "fc7":
        return input_parser_fc7
    elif FLAGS.input_mode == "raw":
        return input_parser_raw


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
            size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
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
            size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
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

    if FLAGS.input_mode == "fc7":
        return lambda dataset: dataset
    elif FLAGS.input_mode == "raw":
        return transformation_parser


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
