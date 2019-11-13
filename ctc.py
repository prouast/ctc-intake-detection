"""Using CTC for detection of events."""

import tensorflow as tf

def _collapse_sequences(labels, seq_length, def_val, pad_val, replace_with_def, pos):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 1) and
            sequences of interest (e.g., [1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 1]).
        seq_length: The length of each sequence
        def_val: The value which denotes the default value
        pad_val: The value which is used to pad sequences at the end
        replace_with_def: If true, collapsed values are replaced with def_val.
            If false, collapsed values are removed (Tensor is padded
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

    if replace_with_def:

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

def _dense_to_sparse(input, eos_token=-1):
    """Convert dense tensor to sparse"""
    idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
    values = tf.gather_nd(input, idx)
    shape = tf.shape(input, out_type=tf.int64)
    sparse = tf.SparseTensor(idx, values, shape)
    return sparse

@tf.function
def _ctc_loss_selective_collapse(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """Collapse repeated non-default values in labels, but keep default values.
        This should force the network to learn outputting either only one event value in a row.
        For inference, epsilons should be replaced with def_val"""
    # Collapse repeated non-def_val's in labels without replacing
    labels, _ = _collapse_sequences(labels, seq_length,
        def_val=def_val, pad_val=pad_val, replace_with_def=False, pos=pos)
    # CTC loss with selectively collapsed labels
    seq_lengths = tf.fill([batch_size], seq_length)
    loss = tf.compat.v1.nn.ctc_loss(
        labels=_dense_to_sparse(labels, eos_token=-1),
        inputs=logits,
        sequence_length=seq_lengths,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=False,
        time_major=False)
    # Reduce loss to scalar
    return tf.reduce_mean(loss)

@tf.function
def _ctc_loss_all_collapse(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """Collapse all repeated values as part of ctc loss."""
    # CTC loss with all collapsed labels
    seq_lengths = tf.fill([batch_size], seq_length)
    loss = tf.compat.v1.nn.ctc_loss(
        labels=_dense_to_sparse(labels, eos_token=-1),
        inputs=logits,
        sequence_length=seq_lengths,
        preprocess_collapse_repeated=True,
        ctc_merge_repeated=False,
        time_major=False)
    # Reduce loss to scalar
    return tf.reduce_mean(loss)

@tf.function
def _ctc_loss_naive(labels, logits, batch_size, seq_length):
    """Naive CTC loss
    This loss only considers the probability of the single path
        implied by the labels without any collapsing. Loss is computed as the
        negative log likelihood of the probability.
    """
    logits = tf.nn.softmax(logits)
    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1])
    # Reduce num_classes by getting indexes that should have high logits
    flat_idx = flat_labels + tf.cast(tf.range(tf.shape(logits)[0] * \
        tf.shape(logits)[1]) * tf.shape(logits)[2], tf.int32)
    loss = tf.reshape(tf.gather(flat_logits, flat_idx), [batch_size, seq_length])
    # Reduce seq_length by negative log-likelihood of product
    loss = tf.reduce_sum(tf.negative(tf.math.log(loss)), axis=1)
    # Reduce mean of batch losses
    loss = tf.reduce_mean(loss)
    return loss

def ctc_loss(labels, logits, ctc_mode, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """Return ctc loss corresponding to ctc_mode"""
    if ctc_mode == 'naive':
        return _ctc_loss_naive(labels, logits,
            batch_size=batch_size, seq_length=seq_length)
    elif ctc_mode == 'selective_collapse':
        return _ctc_loss_selective_collapse(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=0, pad_val=-1)
    elif ctc_mode == 'all_collapse':
        return _ctc_loss_all_collapse(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=0, pad_val=-1)

@tf.function
def _greedy_decode_and_collapse(inputs, num_classes, seq_length, use_epsilon=True, def_val=0, pad_val=-1, pos='middle'):
    """Naive inference by retrieving most likely output at each time-step.

    Args:
        inputs: The prediction in form of logits. [samples, time_steps, num_classes]
        num_classes: The number of classes considered in prediction.
        seq_length: Sequence length for each item in inputs.
        use_epsilon: If yes, contains an extra class for CTC epsilon at the last index
        pos: How should predictions (excluding 0) be collapsed to 0?
    Returns:
        Tuple
        * the decoded sequence [seq_length]
        * indices of 1 predictions
    """
    def greedy_decode(input):
        cat_ids = tf.cast(tf.argmax(input, axis=1), tf.int32)
        if use_epsilon:
            cat_ids = tf.where(tf.equal(cat_ids, num_classes-1),
                tf.zeros([seq_length], tf.int32), cat_ids) # Set epsilons to 0
        row_ids = tf.range(tf.shape(input)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, cat_ids], axis=1)
        return idx[:,1]
    decoded = tf.map_fn(greedy_decode, inputs, dtype=tf.int32)
    collapsed, _ = _collapse_sequences(decoded, seq_length, def_val=def_val,
        pad_val=pad_val, replace_with_def=True, pos=pos)
    # Return collapsed output and indices of ones
    return collapsed, decoded

def ctc_decode_logits(logits, ctc_mode, num_classes, seq_length):
    """Decode ctc logits corresponding to ctc_mode"""
    if ctc_mode == 'naive':
        return _greedy_decode_and_collapse(logits,
            num_classes=num_classes, seq_length=seq_length, use_epsilon=False)
    elif ctc_mode == 'selective_collapse':
        return _greedy_decode_and_collapse(logits,
            num_classes=num_classes, seq_length=seq_length, use_epsilon=True)
    elif ctc_mode == 'all_collapse':
        return _greedy_decode_and_collapse(logits,
            num_classes=num_classes, seq_length=seq_length, use_epsilon=True)
