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
                return tf.math.floordiv(end + start, 2)
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
