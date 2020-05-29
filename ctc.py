"""Using CTC for detection of events."""

import tensorflow as tf
import tensorflow_addons as tfa
import collections
from absl import logging
from tensorflow_ctc_ext_beam_search_decoder import ctc_ext_beam_search_decoder

### Collapse functions

@tf.function
def _collapse_sequences(labels, seq_length, def_val, pad_val, mode, pos):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 0) and
            event sequences of interest (e.g., [0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0]).
        seq_length: The length of each sequence
        def_val: The value which denotes the default value
        pad_val: The value which is used to pad sequences at the end
        mode:
            'collapse_events_replace_collapsed'
                (part of inference)
                - Collapse events and replace collapsed values with def_val.
                    {e.g., 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0}
                => For deriving prediction from decoded logits
            'collapse_all_remove_collapsed'
                (ctc_loss with use_def, computing sequence weight)
                - Collapse event and default sequences, remove collapsed values.
                    {e.g., 0, 1, 0, 2, 0, -1, -1, -1, -1, -1, -1}
                => Prepare labels for ctc loss (with default class)
            'collapse_events_remove_def'
                (ctc_loss without use_def)
                - Collapse event sequences, remove collapsed values and all def.
                    {e.g., 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1}
                => Prepare labels for ctc loss (without default class)
            'collapse_events_remove_collapsed'
                (not used at the moment)
                - Collapse only events and remove collapsed values.
                    {e.g., 0, 0, 1, 0, 0, 2, 0, -1, -1, -1, -1}
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

    if mode == 'collapse_events_replace_collapsed':

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

    elif mode == 'collapse_events_remove_collapsed':

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

    elif mode == 'collapse_events_remove_def':

        # Mask for all def_val in the sequence
        non_default_mask = tf.not_equal(labels, tf.fill(tf.shape(labels), def_val))

        # Combine with mask for all first sequence elements
        mask = tf.logical_and(non_default_mask, prev_mask)
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

    elif mode == 'collapse_all_remove_collapsed':

        # Mask for all sequence starts
        flat_updates = tf.boolean_mask(labels, prev_mask, axis=0)

        # Determine new sequence lengths / max length
        new_seq_len = tf.reduce_sum(tf.cast(prev_mask, tf.int32), axis=1)
        new_maxlen = tf.reduce_max(new_seq_len)

        # Mask idx
        idx_mask = tf.sequence_mask(new_seq_len, maxlen=new_maxlen)
        flat_idx_mask = tf.reshape(idx_mask, [-1])
        idx = tf.range(tf.size(idx_mask))
        flat_idx = tf.boolean_mask(idx, flat_idx_mask, axis=0)
        flat_shape = tf.shape(flat_idx_mask)

        seq_length = new_seq_len

    else:
        raise ValueError("Mode {} not implemented!".format(mode))

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

@tf.function
def _dense_to_sparse(input, eos_token=-1):
    """Convert dense tensor to sparse"""
    idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
    values = tf.gather_nd(input, idx)
    shape = tf.shape(input, out_type=tf.int64)
    sparse = tf.SparseTensor(idx, values, shape)
    return sparse

@tf.function
def _compute_balanced_sequence_weight(labels, seq_length, def_val, pad_val, pos):
    f_labels = tf.reshape(labels,[-1])
    # Classes, idx, and counts
    y, idx, count = tf.unique_with_counts(f_labels)
    # Predicate that indicates whether pad_val are present
    pred = tf.reduce_any(tf.equal(f_labels, pad_val))
    # Number of present pad_val
    pad_count = tf.reduce_sum(tf.cast(tf.equal(f_labels, pad_val), tf.int32))
    # Total count excluding pad_val
    total_count = tf.size(f_labels) - pad_count
    # Label count excluding pad_val
    label_count = tf.cond(pred=pred,
        true_fn=lambda: tf.size(y) - 1, false_fn=lambda: tf.size(y))
    # Calculate weights for labels, set weight for pad_val to 1.0
    pad_index = tf.argmax(tf.cast(tf.equal(y, pad_val), tf.int32), output_type=tf.int32)
    def calc_weight_no_pad(x):
        return tf.divide(tf.divide(total_count, count[x]),
            tf.cast(label_count, tf.float64))
    def calc_weight_with_pad(x):
        return tf.cond(pred=tf.equal(x, pad_index),
                       true_fn=lambda: tf.constant(1.0, tf.float64),
                       false_fn=lambda: calc_weight_no_pad(x))
    def calc_weight(x):
        return tf.cond(pred=pred,
                       true_fn=lambda: calc_weight_with_pad(x),
                       false_fn=lambda: calc_weight_no_pad(x))
    elems = tf.range(0, label_count+1, 1, tf.int32)
    class_weights = tf.map_fn(fn=calc_weight, elems=elems, dtype=tf.float64)
    # Gather sample weights in original tensor structure
    sample_weights = tf.gather(class_weights, idx)
    sample_weights = tf.reshape(sample_weights, tf.shape(labels))
    # Collect weights on sequence level
    sample_weights = tf.reduce_mean(sample_weights, axis=1)
    return tf.cast(sample_weights, tf.float32)

def get_collapse_fn_for_preprocessing(use_def, seq_length, def_val, pad_val):
    def collapse_all_remove_collapsed(labels):
        collapsed, _ = _collapse_sequences(labels, seq_length, def_val=def_val,
            pad_val=pad_val, mode="collapse_all_remove_collapsed", pos='middle')
        return collapsed
    def collapse_events_remove_def(labels):
        collapsed, _ = _collapse_sequences(labels, seq_length, def_val=def_val,
            pad_val=pad_val, mode="collapse_events_remove_def", pos='middle')
        return collapsed
    if use_def:
        return collapse_all_remove_collapsed
    else:
        return collapse_events_remove_def

def collapse_events_replace_collapsed(decoded, seq_length, def_val, pad_val, pos='middle'):
    collapsed, _ = _collapse_sequences(decoded, seq_length, def_val=def_val,
        pad_val=pad_val, mode='collapse_events_replace_collapsed', pos=pos)
    return collapsed

##### Loss functions

@tf.function
def _loss_ctc(labels, logits, batch_size, seq_length, def_val, pad_val, blank_index, training, use_def, pos='middle'):
    """CTC loss
    - Loss: CTC loss (preprocess_collapse_repeated=False, ctc_merge_repeated=True)
    if use_def:
        - Representation: Keep {event_val and def_val}
        - Collapse: Collapse {event_val and def_val} {e.g., 01020-1-1-1}
        # index 1 is default class
    else:
        - Representation: Keep {event_val} only, no default values
        - Collapse: Collapse event_val before loss (pad ends) {e.g., 12-1-1-1}
    # index 0 is blank label
    """
    if training:
        # Compute sequence weights to account for dataset imbalance
        sequence_weights = _compute_balanced_sequence_weight(
            labels, seq_length, def_val, pad_val, pos)
    labels = _dense_to_sparse(labels, eos_token=pad_val)
    logit_lengths = tf.fill([batch_size], seq_length)
    # The loss
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=None,
        logit_length=logit_lengths,
        logits_time_major=False,
        blank_index=blank_index)
    if training:
        # Multiply loss by sequence weights
        loss = sequence_weights * loss
    # Reduce loss to scalar
    return tf.reduce_mean(loss)

@tf.function
def _loss_crossent(labels, logits):
    """Cross-entropy loss"""
    weights = tf.ones_like(labels, dtype=tf.float32)
    # Calculate and scale cross entropy
    loss = tfa.seq2seq.sequence_loss(
        logits=logits,
        targets=tf.cast(labels, tf.int32),
        weights=weights)
    return loss

def loss(labels, labels_c, logits, loss_mode, batch_size, seq_length, def_val, pad_val, blank_index, training, use_def, pos='middle'):
    """Return loss corresponding to loss_mode"""
    if loss_mode == 'ctc':
        return _loss_ctc(labels_c, logits, batch_size=batch_size,
            seq_length=seq_length, def_val=def_val, pad_val=pad_val,
            blank_index=blank_index, training=training, use_def=use_def)
    elif loss_mode == 'crossent':
        return _loss_crossent(labels, logits)

##### Decoding

@tf.function
def _greedy_decode(inputs, seq_length, blank_index, def_val):
    """Naive inference by retrieving most likely output at each time-step.

    Args:
        inputs: The prediction in form of logits. [batch_size, time_steps, num_classes]
        seq_length: The length of the sequences
        blank_index: The index of blank which will be set to def_val (or None)
        def_val: The value associated with the default event
    Returns:
        decoded: The decoded sequence [seq_length]
    """
    # Infer predictions using argmax
    decoded = tf.cast(tf.argmax(inputs, axis=-1), tf.int32)
    if blank_index is not None:
        # Set epsilon_index to def_val
        decoded = tf.where(tf.equal(decoded, blank_index),
            tf.fill([seq_length], def_val), decoded)
    return decoded

@tf.function
def _ctc_decode(inputs, seq_length, blank_index, def_val, use_def, shift):
    """Perform ctc decoding"""
    batch_size = inputs.get_shape()[0]
    # Decode uses time major
    inputs = tf.transpose(a=inputs, perm=[1, 0, 2])
    seq_lengths = tf.fill([batch_size], seq_length)
    # Perform beam search
    indices, values, shape, indices_u, values_u, shape_u, log_probs = ctc_ext_beam_search_decoder(
        inputs=inputs, sequence_length=seq_lengths,
        beam_width=10, blank_index=blank_index, top_paths=1,
        blank_label=0)
    decoded = tf.sparse.SparseTensor(indices[0], values[0], shape[0])
    decoded = tf.cast(tf.sparse.to_dense(decoded), tf.int32)
    decoded_u = tf.sparse.SparseTensor(indices_u[0], values_u[0], shape_u[0])
    decoded_u = tf.cast(tf.sparse.to_dense(decoded_u), tf.int32)
    # Set event vals
    decoded = tf.where(tf.not_equal(decoded, 0), decoded+shift, decoded)
    decoded_u = tf.where(tf.not_equal(decoded_u, 0), decoded_u+shift, decoded_u)
    # Set default vals
    decoded = tf.where(tf.equal(decoded, 0), def_val, decoded)
    decoded_u = tf.where(tf.equal(decoded_u, 0), def_val, decoded_u)
    return decoded_u, decoded

def decode(logits, loss_mode, seq_length, blank_index, def_val, use_def, shift):
    """Decode ctc logits corresponding to loss_mode"""
    if loss_mode == 'ctc':
        return _ctc_decode(logits, seq_length=seq_length,
            blank_index=blank_index, def_val=def_val, use_def=use_def,
            shift=shift)
    elif loss_mode == 'crossent':
        return _greedy_decode(logits, seq_length=seq_length,
            blank_index=blank_index, def_val=def_val)
