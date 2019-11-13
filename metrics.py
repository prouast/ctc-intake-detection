"""Get F1 metric based on precision and recall metrics, which in turn are based
    on true positive, true negative, and false positive."""

import tensorflow as tf

@tf.function
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
        def_val: The default value for non-events
        seq_length: The sequence length.

    Returns:
        tp: True positives (number of true sequences of 1s predicted with at
                least one predicting 1)
        fp_1: False positives type 1 (number of excess predicting 1s matching
                a true sequence of 1s in excess)
        fp_2: False positives type 1 (number number of predicting 1s not
                matching a true sequence of 1s)
        fn: False negatives (number of true sequences of 1s not matched by at
                least one predicting 1)
    """
    def sequence_masks(labels, def_val, seq_length):
        """Generate masks [batch, max_seq_count, seq_length] of all event sequences"""
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
            lambda: tf.fill([batch_size, 0, seq_length], def_val),
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

    # Derive positive ground truth mask reshape to [n_gt_seq, seq_length]
    pos_mask, max_seq_count = sequence_masks(labels, def_val, seq_length)
    pos_mask = tf.reshape(pos_mask, [-1, seq_length])

    # Stack predictions accordingly
    pred_stacked = tf.reshape(tf.tile(tf.expand_dims(predictions, axis=1), [1, max_seq_count, 1]), [-1, seq_length])

    # Remove empty masks
    empty_mask = tf.greater(tf.reduce_sum(pos_mask, axis=1), 0)

    pos_mask = tf.cond(empty,
        lambda: pos_mask,
        lambda: tf.boolean_mask(pos_mask, empty_mask))
    pred_stacked = tf.cond(empty,
        lambda: pred_stacked,
        lambda: tf.boolean_mask(pred_stacked, empty_mask))

    # Calculate number of predictions for pos sequences
    pred_sums = tf.map_fn(
        fn=lambda x: tf.reduce_sum(tf.boolean_mask(x[0], x[1])),
        elems=(pred_stacked, pos_mask), dtype=tf.int32)

    # Calculate true positive, false positive and false negative count
    tp = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 1, lambda: 0), pred_sums))
    fn = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 0, lambda: 1), pred_sums))
    fp_1 = tf.cond(empty,
        lambda: 0,
        lambda: tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 1, lambda: count-1, lambda: 0), pred_sums)))
    fp_2 = tf.reduce_sum(tf.boolean_mask(predictions, mask=neg_mask))

    tp = tf.cast(tp, tf.float32)
    fp_1 = tf.cast(fp_1, tf.float32)
    fp_2 = tf.cast(fp_2, tf.float32)
    fn = tf.cast(fn, tf.float32)

    return tp, fp_1, fp_2, fn

class TP_FP1_FP2_FN(tf.keras.metrics.Metric):
    def __init__(self, def_val, seq_length, name=None, dtype=None):
        super(TP_FP1_FP2_FN, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.def_val = def_val
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp_1 = self.add_weight('total_fp_1',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp_2 = self.add_weight('total_fp_2',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fn = self.add_weight('total_fn',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, fp_1, fp_2, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred,
            def_val=self.def_val, seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp_1.assign_add(fp_1)
        self.total_fp_2.assign_add(fp_2)
        self.total_fn.assign_add(fn)

    def result(self):
        return self.total_tp, self.total_fp_1, self.total_fp_2, self.total_fn

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fp_1.assign(0)
        self.total_fp_2.assign(0)
        self.total_fn.assign(0)

class Precision(tf.keras.metrics.Metric):
    def __init__(self, def_val, seq_length, name=None, dtype=None):
        super(Precision, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.def_val = def_val
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp = self.add_weight('total_fp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, fp_1, fp_2, _ = evaluate_interval_detection(
            labels=y_true, predictions=y_pred,
            def_val=self.def_val, seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp.assign_add(fp_1)
        self.total_fp.assign_add(fp_2)

    def result(self):
        return tf.math.divide_no_nan(
            self.total_tp,
            self.total_tp + self.total_fp)

    def reset_states(self):
        self.total_tp.assign(0)
        self.total_fp.assign(0)

class Recall(tf.keras.metrics.Metric):
    def __init__(self, def_val, seq_length, name=None, dtype=None):
        super(Recall, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.def_val = def_val
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fn = self.add_weight('total_fn',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, _, _, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred,
            def_val=self.def_val, seq_length=self.seq_length)
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
    def __init__(self, def_val, seq_length, name=None, dtype=None):
        super(F1, self).__init__(name=name, dtype=dtype)
        self.seq_length = seq_length
        self.def_val = def_val
        self.total_tp = self.add_weight('total_tp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fn = self.add_weight('total_fn',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)
        self.total_fp = self.add_weight('total_fp',
            shape=(), initializer=tf.zeros_initializer, dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        tp, fp_1, fp_2, fn = evaluate_interval_detection(
            labels=y_true, predictions=y_pred,
            def_val=self.def_val, seq_length=self.seq_length)
        self.total_tp.assign_add(tp)
        self.total_fp.assign_add(fp_1)
        self.total_fp.assign_add(fp_2)
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
