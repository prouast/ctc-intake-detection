"""Using CTC for detection of events."""

import tensorflow as tf
import collections
from absl import logging

NEG_INF = -float("inf")

def _collapse_sequences(labels, seq_length, def_val, pad_val, mode, pos):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 0) and
            sequences of interest (e.g., [0, 0, 1, 1, 1, 0, 0, 2, 2, 2, 0]).
        seq_length: The length of each sequence
        def_val: The value which denotes the default value
        pad_val: The value which is used to pad sequences at the end
        mode:
            'replace_collapsed' - Collapsed values are replaced with def_val.
            'remove_collapsed' - Collapsed values are removed (Tensor is padded
                with pad_val).
            'remove_def' - Remove all def values after collapse (Tensor is
                padded with pad_val).
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

    if mode == 'replace_collapsed':

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

    elif mode == 'remove_collapsed':

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

    elif mode == 'remove_def':

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

def _compute_balanced_sample_weight(labels):
    """Calculate the balanced sample weight for imbalanced data."""
    f_labels = tf.reshape(labels,[-1]) if labels.get_shape().ndims == 2 else labels
    y, idx, count = tf.unique_with_counts(f_labels)
    total_count = tf.size(f_labels)
    label_count = tf.size(y)
    calc_weight = lambda x: tf.divide(tf.divide(total_count, x),
        tf.cast(label_count, tf.float64))
    class_weights = tf.map_fn(fn=calc_weight, elems=count, dtype=tf.float64)
    sample_weights = tf.gather(class_weights, idx)
    sample_weights = tf.reshape(sample_weights, tf.shape(labels))
    return tf.cast(sample_weights, tf.float32)

##### Loss functions

@tf.function
def _loss_ctc_def_all(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """CTC loss
    - Loss: CTC loss (preprocess_collapse_repeated=True, ctc_merge_repeated=True)
    - Representation: Keep {event_val, def_val}
    - Collapse: Collapse all vals before loss
    """
    # CTC loss with all collapsed labels including def_val
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
def _loss_ctc_def_event(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """CTC loss
    - Loss: CTC loss (preprocess_collapse_repeated=False, ctc_merge_repeated=False)
    - Representation: Keep {event_val, def_val}
    - Collapse: Collapse event_val before loss (pad ends)
    """
    # Collapse repeated non-def_val's in labels without replacing
    labels, _ = _collapse_sequences(labels, seq_length,
        def_val=def_val, pad_val=pad_val, mode='remove_collapsed', pos=pos)
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
def _loss_ctc_ndef_all(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """CTC loss
    - Loss: CTC loss (preprocess_collapse_repeated=TODO, ctc_merge_repeated=TODO)
    - Representation: Keep {event_val} only
    - Collapse: Collapse event_val before loss (pad ends)
    # 0 / index 0 is blank label
    """
    # Collapse repeated events in labels, remove all def_val
    labels, label_lengths = _collapse_sequences(labels, seq_length,
        def_val=def_val, pad_val=pad_val, mode='remove_def', pos=pos)
    logit_lengths = tf.fill([batch_size], seq_length)
    loss = tf.nn.ctc_loss(
        labels=_dense_to_sparse(labels, eos_token=-1),
        logits=logits,
        label_length=None,
        logit_length=logit_lengths,
        logits_time_major=False,
        blank_index=0)
    # Reduce loss to scalar
    return tf.reduce_mean(loss)

@tf.function
def _loss_naive_def_none(labels, logits, batch_size, seq_length):
    """Naive CTC loss with no collapse
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

@tf.function
def _loss_naive_def_event(labels, logits, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """Naive CTC loss with event collapse
    This loss only considers the probability of the single path
        implied by the labels after collapsing events to a single element and
        replacing with def_val. Loss is computed as the negative log
        likelihood of the probability.
    """
    # Get weights from non-collapsed labels
    sample_weights = _compute_balanced_sample_weight(tf.reshape(labels, [-1]))
    # Collapse sequences to make lstm learn to predict only once per event
    labels, _ = _collapse_sequences(labels, seq_length,
        def_val=def_val, pad_val=pad_val, mode='replace_collapsed', pos=pos)
    logits = tf.nn.softmax(logits)
    flat_labels = tf.reshape(labels, [-1])
    flat_logits = tf.reshape(logits, [-1])
    # Reduce num_classes by getting indexes that should have high logits
    flat_idx = flat_labels + tf.cast(tf.range(tf.shape(logits)[0] * \
        tf.shape(logits)[1]) * tf.shape(logits)[2], tf.int32)
    flat_loss = tf.gather(flat_logits, flat_idx)
    # Negative log
    flat_loss = tf.negative(tf.math.log(flat_loss))
    # Weigh with balanced sample weight across seq_length
    #sample_weights = _compute_balanced_sample_weight(flat_labels)
    flat_loss = tf.multiply(flat_loss, sample_weights)
    loss = tf.reshape(flat_loss, [batch_size, seq_length])
    # Reduce seq_length by sum
    loss = tf.reduce_sum(loss, axis=1)
    # Reduce mean of batch losses
    return tf.reduce_mean(loss)

def loss(labels, logits, loss_mode, batch_size, seq_length, def_val, pad_val, pos='middle'):
    """Return loss corresponding to loss_mode"""
    if loss_mode == 'ctc_def_all':
        return _loss_ctc_def_all(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=def_val, pad_val=pad_val)
    elif loss_mode == 'ctc_def_event':
        return _loss_ctc_def_event(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=def_val, pad_val=pad_val)
    elif loss_mode == 'ctc_ndef_all':
        return _loss_ctc_ndef_all(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=def_val, pad_val=pad_val)
    elif loss_mode == 'naive_def_none':
        return _loss_naive_def_none(labels, logits,
            batch_size=batch_size, seq_length=seq_length)
    elif loss_mode == 'naive_def_event':
        return _loss_naive_def_event(labels, logits,
            batch_size=batch_size, seq_length=seq_length, def_val=def_val, pad_val=pad_val)

##### Decoding

@tf.function
def _greedy_decode(inputs, seq_length, def_index, epsilon_index, shift=0, def_val=0):
    """Naive inference by retrieving most likely output at each time-step.

    Args:
        inputs: The prediction in form of logits. [batch_size, time_steps, num_classes]
        seq_length: The length of the sequences
        def_index: The index of the default event which will be set to def_val (or None)
        epsilon_index: The index of epsilon which will be set to def_val (or None)
        shift: How far are events' indices from their values
        def_val: The value associated with the default event
    Returns:
        decoded: The decoded sequence [seq_length]
    """
    # Infer predictions using argmax
    decoded = tf.cast(tf.argmax(inputs, axis=-1), tf.int32)
    decoded = tf.add(decoded, shift)
    if def_index is not None:
        # Set def_index to def_val
        decoded = tf.where(tf.equal(decoded, def_index-shift),
            tf.fill([seq_length], def_val), decoded)
    if epsilon_index is not None:
        # Set epsilon_index to def_val
        decoded = tf.where(tf.equal(decoded, epsilon_index-shift),
            tf.fill([seq_length], def_val), decoded)
    return decoded

@tf.function
def _greedy_decode_and_collapse(inputs, seq_length, def_index, epsilon_index, shift=0, def_val=0, pad_val=-1, pos='middle'):
    """Retrieve most likely output at each time-step and collapse predictions."""
    decoded = _greedy_decode(inputs, seq_length=seq_length, def_index=def_index,
        epsilon_index=epsilon_index, shift=shift, def_val=def_val)
    collapsed, _ = _collapse_sequences(decoded, seq_length, def_val=def_val,
        pad_val=pad_val, mode='replace_collapsed', pos=pos)
    return collapsed, decoded

class Beam:
    def __init__(self, p_b, p_nb, seq):
        self.p_b = p_b
        self.p_nb = p_nb
        self.seq = seq
        self.bu_seq_nb = tf.constant([], tf.int32)
        self.bu_seq_b = tf.constant([], tf.int32)
        self.bu_seq_nb_cand = [(NEG_INF, self.bu_seq_nb)] # list of bu_seq_nb merging candidates for current step with probabilities
        self.bu_seq_b_cand = [(NEG_INF, self.bu_seq_b)] # list of bu_seq_nb merging candidates for current step with probabilities
    def __str__(self):
        return  "Beam [seq = %s, p_b = %s, p_nb = %s, bu_seq_b = %s, bu_seq_nb = %s]" \
            % (self.seq, self.p_b, self.p_nb, self.bu_seq_b, self.bu_seq_nb)
    def get_p(self):
        return tf.reduce_logsumexp([self.p_b, self.p_nb])

# TODO: Write efficient tf graph version
def _ctc_decode(inputs, beam_width=10, def_val=-1):
    """Decode with ctc beam search"""
    seq_length, num_events = inputs.shape
    blank = num_events - 1

    # Store the beam entries here
    beams = [Beam(0.0, NEG_INF, tf.constant([], tf.int32))]

    # For each sequence step
    for t in range(seq_length):
        #if t % 1000 == 0:
        #    logging.info("CTC inference step {0}...".format(t))
        def _make_new_beams():
          fn = lambda : Beam(NEG_INF, NEG_INF, [])
          return collections.defaultdict(fn)
        new_beams = _make_new_beams()

        # For all existing beams
        for beam in beams:

            # A. Enter this beam into new proposals if not already there
            new_beam = new_beams[str(beam.seq)]
            new_beam.seq = beam.seq

            # A. 1) Case of non-empty beam with repeated last event
            if tf.size(beam.seq) > 0:
                new_beam.p_nb = tf.reduce_logsumexp([new_beam.p_nb,
                    beam.p_nb + inputs[t, beam.seq[-1]]])
                new_beam.bu_seq_nb_cand.append((
                    tf.reduce_logsumexp([beam.p_nb + inputs[t, beam.seq[-1]]]),
                    tf.concat([beam.bu_seq_nb, [beam.seq[-1]]], 0)))

            # A. 2) Case of adding a blank event
            new_beam.p_b = tf.reduce_logsumexp([new_beam.p_b,
                beam.p_b + inputs[t, blank], beam.p_nb + inputs[t, blank]])
            new_beam.bu_seq_b_cand.append((
                tf.reduce_logsumexp([beam.p_b + inputs[t, blank]]),
                tf.concat([beam.bu_seq_b, [def_val]], 0)))
            new_beam.bu_seq_b_cand.append((
                tf.reduce_logsumexp([beam.p_nb + inputs[t, blank]]),
                tf.concat([beam.bu_seq_nb, [def_val]], 0)))

            # B. Extend this beam with a non-blank event
            for event in range(num_events-1):

                # Enter beam with the new prefix
                new_seq = tf.concat([beam.seq, [event]], 0)
                new_beam = new_beams[str(new_seq)]
                new_beam.seq = new_seq

                # B. 1) Case of repeated event at the end in prefix
                if tf.size(beam.seq) > 0 and beam.seq[-1] == event:
                    # Only consider seqs ending with blank event
                    new_beam.p_nb = tf.reduce_logsumexp([new_beam.p_nb,
                        beam.p_b + inputs[t, event]])
                    new_beam.bu_seq_nb_cand.append((
                        tf.reduce_logsumexp([beam.p_b + inputs[t, event]]),
                        tf.concat([beam.bu_seq_b, [event]], 0)))
                # B. 2) Case of no repeated event
                else:
                    new_beam.p_nb = tf.reduce_logsumexp([new_beam.p_nb,
                        beam.p_b + inputs[t, event], beam.p_nb + inputs[t, event]])
                    new_beam.bu_seq_nb_cand.append((
                        tf.reduce_logsumexp([beam.p_b + inputs[t, event]]),
                        tf.concat([beam.bu_seq_b, [event]], 0)))
                    new_beam.bu_seq_nb_cand.append((
                        tf.reduce_logsumexp([beam.p_nb + inputs[t, event]]),
                        tf.concat([beam.bu_seq_nb, [event]], 0)))

        # Sort and trim the beam at the end of each sequence step
        beams = sorted(new_beams.values(),
            key=lambda x: x.get_p(), reverse=True)
        beams = beams[:beam_width]

        # Resolve the most likely bu_prefix to each beam.
        for beam in beams:
            beam.bu_seq_nb = sorted(beam.bu_seq_nb_cand, key=lambda x: x[0], reverse=True)[0][1]
            beam.bu_seq_b = sorted(beam.bu_seq_b_cand, key=lambda x: x[0], reverse=True)[0][1]

    best = beams[0]
    bu_seq = best.bu_seq_nb if best.p_nb > best.p_b else best.bu_seq_b
    bu_seq = tf.add(bu_seq, 1)

    # Pad the prefix to seq_length
    paddings = [[0, seq_length-tf.shape(best.seq)[0]]]
    seq = tf.pad(best.seq + 1, paddings, 'CONSTANT', constant_values=def_val)

    return bu_seq, seq

@tf.function
def _ctc_decode_batch(inputs, beam_width, seq_length, def_val=0, pad_val=-1):
    # Add empty batch dimension if needed
    inputs = tf.cond(
        pred=tf.equal(tf.size(tf.shape(inputs)), 2),
        true_fn=lambda: tf.expand_dims(inputs, 0),
        false_fn=lambda: tf.identity(inputs))
    # Decode each example
    decoded, seq = tf.map_fn(
        fn=lambda x: _ctc_decode(x, beam_width=beam_width, def_val=-1),
        elems=inputs, dtype=(tf.int32, tf.int32))
    # Collapse sequences in case there are any
    collapsed, _ = _collapse_sequences(decoded, seq_length, def_val=0,
        pad_val=-1, mode='replace_collapsed', pos='middle')
    return collapsed, seq

def decode_logits(logits, loss_mode, seq_length, num_event):
    """Decode ctc logits corresponding to loss_mode"""

    if loss_mode == 'ctc_def_all':
        return _greedy_decode_and_collapse(logits, seq_length=seq_length,
            def_index=0, epsilon_index=num_event+1)
    elif loss_mode == 'ctc_def_event':
        return _greedy_decode_and_collapse(logits, seq_length=seq_length,
            def_index=0, epsilon_index=num_event+1)
    elif loss_mode == 'ctc_ndef_all':
        return _ctc_decode_batch(logits, beam_width=5, seq_length=seq_length)
    elif loss_mode == 'naive_def_none':
        return _greedy_decode_and_collapse(logits, seq_length=seq_length,
            def_index=0, epsilon_index=None)
    elif loss_mode == 'naive_def_event':
        return _greedy_decode_and_collapse(logits, seq_length=seq_length,
            def_index=0, epsilon_index=None)
