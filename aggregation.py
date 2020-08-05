import tensorflow as tf

def argmax(input):
  """When evaluating votes, do choose idle in case of a tie"""
  def last_max_index(input):
    max = tf.reduce_max(input)
    idx = tf.where(tf.equal(input, max))
    return idx[0]
  idxs = tf.map_fn(lambda x: last_max_index(x), input)
  return idxs

class ConcatAggregator():
  def __init__(self, n, idx):
    self.agg = None
    self.idx = idx
    self.n = n

  def step(self, i, b_input):
    if i == 0:
      # First batch
      self.agg = b_input[:,:self.idx+1]
    elif i == self.n-1:
      # Last batch
      self.agg = tf.concat([self.agg, b_input[:,self.idx:]], 1)
    else:
      # Middle batch
      self.agg = tf.concat([self.agg, tf.expand_dims(b_input[:,self.idx], 0)], 1)

  def result(self):
    return self.agg

class AverageAggregator():
  def __init__(self, num_classes, seq_length):
    self.sum = None
    self.num = None
    self.seq_length = seq_length
    self.num_classes = num_classes

  def step(self, i, b_input):
    # Construct step vals
    sum_incr = b_input
    num_incr = tf.ones([1, self.seq_length, self.num_classes], tf.float32)
    if i == 0:
      self.sum = sum_incr
      self.num = num_incr
    else:
      empty_row = tf.zeros([1, 1, self.num_classes], tf.float32)
      empty_prev_rows = tf.zeros([1, i, self.num_classes], tf.float32)
      self.sum = tf.add(
        tf.concat([self.sum, empty_row], 1),
        tf.concat([empty_prev_rows, sum_incr], 1))
      self.num = tf.add(
        tf.concat([self.num, empty_row], 1),
        tf.concat([empty_prev_rows, num_incr], 1))

  def result(self):
    return tf.math.divide(self.sum, self.num)

class VideoLevelPredsAggregator():
  def __init__(self, logits_aggregator, decode_fn):
    self.decode_fn = decode_fn
    self.logits_aggregator = logits_aggregator

  def step(self, i, b_inputs):
    self.logits_aggregator.step(i, b_inputs)

  def result(self):
    """Decode predictions on video level"""
    logits = self.logits_aggregator.result()
    preds, _ = self.decode_fn(logits)
    return preds

class BatchLevelConcatPredsAggregator():
  def __init__(self, n, idx, decode_fn):
    self.preds = None
    self.n = n
    self.decode_fn = decode_fn
    self.idx = idx

  def step(self, i, b_inputs):
    # Decode on batch level
    b_preds, _ = self.decode_fn(b_inputs)
    # Select prediction for one index
    if i == 0:
      # First batch
      self.preds = b_preds[0,:self.idx+1]
    elif i == self.n-1:
      # Last batch
      self.preds = tf.concat([self.preds, b_preds[0,self.idx:]], 0)
    else:
      # Middle batch
      self.preds = tf.concat([self.preds, b_preds[:,self.idx]], 0)

  def result(self, logits=None):
    """Return labels, logits, preds"""
    return tf.expand_dims(self.preds, 0)

class BatchLevelVotedPredsAggregator():
  def __init__(self, num_classes, seq_length, def_val, decode_fn):
    self.pred_votes = None
    self.decode_fn = decode_fn
    self.def_val = def_val
    self.seq_length = seq_length
    self.num_classes = num_classes

  def step(self, i, b_inputs):
    # Decode on batch level
    b_preds, _ = self.decode_fn(b_inputs)
    # Transform from representation to [0, 1, ...]
    b_preds -= self.def_val
    # Each sliding batch casts votes for included indices
    # Construct preds tensor
    if i == 0:
      self.pred_votes = tf.zeros(
        [self.seq_length, self.num_classes], tf.int32)
    else:
      self.pred_votes = tf.concat(
        [self.pred_votes, tf.zeros([1, self.num_classes], tf.int32)], 0)
    # Add votes
    preds_incr = tf.concat([
      tf.zeros([i, self.num_classes], tf.int32),
      tf.one_hot(b_preds[0], depth=self.num_classes, dtype=tf.int32)], 0)
    self.pred_votes += preds_incr

  def result(self):
    """Return labels, logits, preds"""
    preds = tf.cast(argmax(tf.cast(self.pred_votes, tf.int64)), tf.int32)
    # Transform from [0, 1, ...] to representation
    preds += self.def_val
    return tf.expand_dims(preds, 0)
