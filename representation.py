import tensorflow as tf
import tensorflow_addons as tfa
from ctc import ctc_decode
from ctc import greedy_decode

@tf.function
def balanced_sample_weight(labels, maxlen):
  # Sort
  labels = tf.sort(labels, axis=-1, direction='DESCENDING')
  # Reduce each label combination to unique sequence
  def reduce_concat(input, maxlen):
    maxlen = input.shape[-1] if maxlen is None else maxlen
    dec = 10**tf.range(maxlen - 1, -1, -1)
    return tf.reduce_sum(input * dec, axis=-1)
  labels = reduce_concat(labels, maxlen)
  # Identify unique label combinations, idx, and counts
  y, idx, count = tf.unique_with_counts(labels)
  # Calculate class weights
  total_count = tf.size(labels)
  label_count = tf.size(y)
  calc_weight = lambda x: tf.divide(tf.divide(total_count, x),
    tf.cast(label_count, tf.float64))
  class_weights = tf.map_fn(fn=calc_weight, elems=count, dtype=tf.float64)
  # Gather sample weights
  sample_weights = tf.gather(class_weights, idx)
  return tf.cast(sample_weights, tf.float32)

@tf.function
def collapse_sequences(labels, seq_length, def_val, pad_val, mode, pos):
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

  return result, seq_length, new_maxlen

@tf.function
def dense_to_sparse(input, eos_token=-1):
  """Convert dense tensor to sparse"""
  idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
  values = tf.gather_nd(input, idx)
  shape = tf.shape(input, out_type=tf.int64)
  sparse = tf.SparseTensor(idx, values, shape)
  return sparse

class Representation():

  def __init__(self, blank_index, def_val, loss_mode, num_event_classes, pad_val, use_def, decode_fn, beam_width):
    # Number of event classes excluding default (idle) event
    self.num_event_classes = num_event_classes
    # Value for default (idle) event
    self.def_val = def_val
    # Is default event used for representation?
    self.use_def = use_def
    # Number of default event classes
    num_def_classes = 1 if use_def else 0
    # Number of blank classes for loss
    self.loss_mode = loss_mode
    num_blank_classes = 1 if self.loss_mode == "ctc" else 0
    # Number of classes for representation during ctc loss
    self.num_classes = num_event_classes + num_def_classes + num_blank_classes
    # Shift required to convert between [1, 2, ...] to chosen representation
    self.shift = def_val - 1 if use_def else def_val
    # Value used for padding the end of collapsed labels
    self.pad_val = pad_val
    # Index used to represent blank event
    self.blank_index = blank_index
    # Decode function
    self.decode_fn = decode_fn
    # Beam width
    self.beam_width = beam_width

  def set_seq_length(self, seq_length):
    self.seq_length = seq_length

  def get_seq_length(self):
    return self.seq_length

  def get_loss_collapse_fn(self):
    def collapse_all_remove_collapsed(labels):
      return collapse_sequences(labels, self.seq_length, def_val=self.def_val,
        pad_val=self.pad_val, mode="collapse_all_remove_collapsed", pos='middle')
    def collapse_events_remove_def(labels):
      return collapse_sequences(labels, self.seq_length, def_val=self.def_val,
        pad_val=self.pad_val, mode="collapse_events_remove_def", pos='middle')
    if self.use_def:
      return collapse_all_remove_collapsed
    else:
      return collapse_events_remove_def

  def get_inference_collapse_fn(self, seq_length=None):
    seq_length = self.seq_length if seq_length is None else seq_length
    def collapse_events_replace_collapsed(decoded):
      collapsed, _, _ = collapse_sequences(decoded, seq_length,
        def_val=self.def_val, pad_val=self.pad_val,
        mode='collapse_events_replace_collapsed', pos='middle')
      return collapsed
    return collapse_events_replace_collapsed

  def get_loss_fn(self, batch_size):
    def ctc_loss_fn(labels, labels_c, logits, label_l):
      """CTC loss via tf.nn.ctc_loss"""
      # Adjust labels_c to def_val
      labels_c = tf.where(tf.not_equal(labels_c, 0),
        labels_c - self.shift, labels_c)
      # Assert labels_c between 0 and num_event_classes
      tf.debugging.assert_less_equal(labels_c, self.num_event_classes)
      tf.debugging.assert_greater_equal(labels_c, 0)
      # Length of logits
      logit_l = tf.fill([batch_size], self.seq_length)
      # Compute loss
      loss = tf.nn.ctc_loss(
        labels=labels_c,
        logits=logits,
        label_length=label_l,
        logit_length=logit_l,
        logits_time_major=False,
        blank_index=self.blank_index)
      # Compute weights
      weights = balanced_sample_weight(labels_c, None)
      return tf.reduce_mean(weights * loss)
    def crossent_loss_fn(labels, labels_c, logits, label_l):
      """Cross-entropy loss"""
      # Adjust labels to def_val
      labels = tf.cast(labels - self.def_val, tf.int32)
      # Assert labels between 0 and num_event_classes
      tf.debugging.assert_less_equal(labels, self.num_event_classes)
      tf.debugging.assert_greater_equal(labels, 0)
      # Calculate cross entropy
      if self.seq_length == 1:
        # Standard loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.squeeze(labels, axis=1),
          logits=tf.squeeze(logits, axis=1))
        loss = tf.cast(loss, tf.float32)
      else:
        # Sequence loss
        seq_weights = tf.ones_like(labels, dtype=tf.float32)
        loss = tfa.seq2seq.sequence_loss(
          logits=logits,
          targets=labels,
          weights=seq_weights,
          average_across_timesteps=True,
          average_across_batch=False)
      batch_weights = balanced_sample_weight(labels_c, None)
      return tf.reduce_mean(batch_weights * loss)
    if self.loss_mode == "ctc":
      # Make sure seq_length is not 1
      assert self.seq_length > 1, "seq_length must be greater than 1"
      return ctc_loss_fn
    else:
      return crossent_loss_fn

  def get_decode_fn(self, batch_size, seq_length=None):
    seq_length = self.seq_length if seq_length is None else seq_length
    def ctc_decode_fn(logits):
      return ctc_decode(logits, batch_size=batch_size, seq_length=seq_length,
        blank_index=self.blank_index, def_val=self.def_val, shift=self.shift,
        beam_width=self.beam_width)
    def greedy_decode_fn(logits):
      return greedy_decode(logits, seq_length=seq_length,
        blank_index=self.blank_index, def_val=self.def_val, shift=self.shift)
    if self.decode_fn == "greedy":
      return greedy_decode_fn
    else:
      return ctc_decode_fn

  def get_event_classes_range(self):
    return range(self.def_val + 1, self.def_val + self.num_event_classes + 1)

  def get_all_classes_range(self):
    return range(self.def_val, self.def_val + self.num_event_classes + 1)

  def get_num_classes(self):
    return self.num_classes

  def get_num_event_classes(self):
    return self.num_event_classes

  def get_def_val(self):
    return self.def_val

  def get_shift(self):
    return self.shift

  def get_loss_mode(self):
    return self.loss_mode
