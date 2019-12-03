"""Get F1 metric based on precision and recall metrics, which in turn are based
    on true positive, true negative, and false positive."""

import tensorflow as tf

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
    def __init__(self, event_val, def_val, seq_length, other_vals=[], name=None, dtype=None):
        super(TP_FP1_FP2_FP3_FN, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.event_val = event_val
        self.def_val = def_val
        self.other_vals = other_vals
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
        tp, fp_1, fp_2, fp_3, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred, event_val=self.event_val,
            def_val=self.def_val, other_vals=self.other_vals,
            seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp_1.assign_add(fp_1)
        self.total_fp_2.assign_add(fp_2)
        self.total_fp_3.assign_add(tf.reduce_sum(fp_3))
        self.total_fn.assign_add(fn)

    def result(self):
        return self.total_tp, self.total_fp_1, self.total_fp_2, self.total_fp_3, self.total_fn

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fp_1.assign(0)
        self.total_fp_2.assign(0)
        self.total_fp_3.assign(0)
        self.total_fn.assign(0)

class Precision(tf.keras.metrics.Metric):
    def __init__(self, event_val, def_val, seq_length, other_vals=[], name=None, dtype=None):
        super(Precision, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.event_val = event_val
        self.def_val = def_val
        self.other_vals = other_vals
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp = self.add_weight('total_fp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, fp_1, fp_2, fp_3, _ = evaluate_interval_detection(
            labels=y_true, predictions=y_pred, event_val=self.event_val,
            def_val=self.def_val, other_vals=self.other_vals,
            seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp.assign_add(fp_1)
        self.total_fp.assign_add(fp_2)
        self.total_fp.assign_add(tf.reduce_sum(fp_3))

    def result(self):
        return tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + self.total_fp)

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fp.assign(0)

class Recall(tf.keras.metrics.Metric):
    def __init__(self, event_val, def_val, seq_length, other_vals=[], name=None, dtype=None):
        super(Recall, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.event_val = event_val
        self.def_val = def_val
        self.other_vals = other_vals
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fn = self.add_weight('total_fn',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, _, _, _, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred, event_val=self.event_val,
            def_val=self.def_val, other_vals=self.other_vals,
            seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fn.assign_add(fn)

    def result(self):
        return tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + self.total_fn)

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fn.assign(0)

class F1(tf.keras.metrics.Metric):
    def __init__(self, event_val, def_val, seq_length, other_vals=[], name=None, dtype=None):
        super(F1, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.event_val = event_val
        self.def_val = def_val
        self.other_vals = other_vals
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fn = self.add_weight('total_fn',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp = self.add_weight('total_fp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, fp_1, fp_2, fp_3, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred, event_val=self.event_val,
            def_val=self.def_val, other_vals=self.other_vals,
            seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp.assign_add(fp_1)
        self.total_fp.assign_add(fp_2)
        self.total_fp.assign_add(tf.reduce_sum(fp_3))
        self.total_fn.assign_add(fn)

    def result(self):
        pre = tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + self.total_fp)
        rec = tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + self.total_fn)
        return tf.where(tf.greater(self.total_tp, 0),
                tf.divide(
                    2 * pre * rec,
                    pre + rec),
                0)

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fn.assign(0)
        self.total_fp.assign(0)
