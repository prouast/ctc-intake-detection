"""Using CTC for detection of events."""

import tensorflow as tf


def _collapse_sequences(labels, seq_length, def_val=1, pad_val=0, replace_with_idle=True, pos='middle'):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 1) and
            sequences of interest (e.g., [1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 1]).
        def_val: The value which denotes the default value
        pad_val: The value which is used to pad sequences at the end
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
    collapsed, _ = _collapse_sequences(decoded, seq_length, def_val=0, pad_val=-1,
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
