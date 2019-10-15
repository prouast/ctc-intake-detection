"""Get F1 metric based on precision and recall metrics, which in turn are based
    on true positive, true negative, and false positive."""

import tensorflow as tf


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


def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across towers."""
    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution, '_outer_control_flow_context'):
            # pylint: disable=protected-access
            if distribution._outer_control_flow_context is None:
                with tf.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            tf.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return tf.contrib.distribute.get_tower_context().merge_call(fn, *args)


def _aggregate_variable(v, collections):
    f = lambda distribution, value: distribution.read_var(value)
    if tf.__version__ < "1.13.1":
        return _aggregate_across_towers(collections, f, v)
    else:
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

        if tf.__version__ < "1.13.1":
            pre = _aggregate_across_towers(
                metrics_collections, lambda _, tp, fp: compute_precision(tp, fp, 'value'), tp, fp)
            rec = _aggregate_across_towers(
                metrics_collections, lambda _, tp, fn: compute_recall(tp, fn, 'value'), tp, fn)
        else:
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

        if tf.__version__ < "1.13.1":
            f1 = _aggregate_across_towers(
                metrics_collections, lambda _, pre, rec: compute_f1(pre, rec, 'value'), pre, rec)
        else:
            f1 = _aggregate_across_replicas(
                metrics_collections, lambda _, pre, rec: compute_f1(pre, rec, 'value'), pre, rec)

        f1_update_op = compute_f1(pre_update_op, rec_update_op, 'update_op')

        if updates_collections:
            tf.add_to_collections(updates_collections, f1_update_op)

        return f1, f1_update_op
