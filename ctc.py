"""Using CTC for detection of events."""

import tensorflow as tf
from tensorflow_ctc_ext_beam_search_decoder import ctc_ext_beam_search_decoder

@tf.function
def greedy_decode(inputs, seq_length, blank_index, def_val, shift):
  """Naive inference by retrieving most likely output at each time-step.

  Args:
    inputs: The prediction in form of logits. [batch_size, time_steps, num_classes]
    seq_length: The length of the sequences
    blank_index: The index of blank which will be set to def_val (or None)
    def_val: The value associated with the default event
    shift: Necessary shift to convert to representation
  Returns:
    decoded: The decoded sequence [seq_length]
  """
  # Infer predictions using argmax
  decoded = tf.cast(tf.argmax(inputs, axis=-1), tf.int32)
  # Adjust event vals according to representation
  decoded = tf.where(tf.not_equal(decoded, 0), decoded+shift, decoded)
  # Set default vals
  decoded = tf.where(tf.equal(decoded, 0), def_val, decoded)
  return decoded, None

@tf.function
def ctc_decode(inputs, batch_size, seq_length, blank_index, def_val, shift, beam_width=10):
  """Perform ctc decoding"""
  # Decode uses time major
  inputs = tf.transpose(a=inputs, perm=[1, 0, 2])
  seq_lengths = tf.fill([batch_size], seq_length)
  # Perform beam search
  indices, values, shape, indices_u, values_u, shape_u, log_probs = ctc_ext_beam_search_decoder(
    inputs=inputs, sequence_length=seq_lengths,
    beam_width=beam_width, blank_index=blank_index, top_paths=1,
    blank_label=0)
  decoded = tf.sparse.SparseTensor(indices[0], values[0], shape[0])
  decoded = tf.cast(tf.sparse.to_dense(decoded), tf.int32)
  decoded_u = tf.sparse.SparseTensor(indices_u[0], values_u[0], shape_u[0])
  decoded_u = tf.cast(tf.sparse.to_dense(decoded_u), tf.int32)
  # Adjust event vals according to representation
  decoded = tf.where(tf.not_equal(decoded, 0), decoded+shift, decoded)
  decoded_u = tf.where(tf.not_equal(decoded_u, 0), decoded_u+shift, decoded_u)
  # Set default vals
  decoded = tf.where(tf.equal(decoded, 0), def_val, decoded)
  decoded_u = tf.where(tf.equal(decoded_u, 0), def_val, decoded_u)
  # We know the shape pf decoded_u, and first dim for decoded
  decoded_u.set_shape([batch_size, seq_length])
  decoded = tf.reshape(decoded, [batch_size, -1])
  return decoded_u, decoded
