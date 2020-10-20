from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import aggregation
from metrics import PredMetrics
from representation import Representation
from metrics import evaluate_interval_detection

BLANK_INDEX = 0
DEF_VAL = 1
PAD_VAL = 0

FLAGS = flags.FLAGS
flags.DEFINE_integer(name='beam_width',
  default=10, help='Beam width to use.')
flags.DEFINE_enum(name='decode_fn',
  default='beam_search', enum_values=["greedy", "beam_search"],
  help='Select the decode fn')
flags.DEFINE_string(name='logits_dir',
  default='predict/logits', help='Directory for logits tfrecord files.')
flags.DEFINE_integer(name='num_classes',
  default=2, help='Number of classes in saved logits.')
flags.DEFINE_string(name='predict_dir',
  default='predict', help='Directory for prediction export.')
flags.DEFINE_integer(name='seq_length',
  default=60, help='Sequence length of saved logits.')

def parse(serialized_example):
  features = tf.io.parse_single_example(
    serialized_example, {
      'example/logits': tf.io.FixedLenFeature([1, FLAGS.seq_length, FLAGS.num_classes], dtype=tf.float32),
      'example/labels': tf.io.FixedLenFeature([1, FLAGS.seq_length], dtype=tf.int64)
  })
  return features['example/logits'], tf.cast(features['example/labels'], tf.int32)

def main(arg=None):
  # Make target dir
  export_dir = os.path.join(FLAGS.predict_dir, "beam_width_" + str(FLAGS.beam_width))
  if not os.path.exists(export_dir):
    os.makedirs(export_dir)
  # Get representation and metrics
  seq_length = FLAGS.seq_length
  num_classes = FLAGS.num_classes
  rep = Representation(blank_index=BLANK_INDEX, def_val=DEF_VAL,
    loss_mode=None, num_event_classes=num_classes-1,
    pad_val=PAD_VAL, use_def=False, decode_fn=FLAGS.decode_fn,
    beam_width=FLAGS.beam_width)
  rep.set_seq_length(seq_length)
  metrics = PredMetrics(rep)
  # Find files
  filenames = sorted(gfile.Glob(os.path.join(FLAGS.logits_dir, "*.tfrecord")))
  # For each file
  for filename in filenames:
    # Get video id
    video_id = os.path.splitext(os.path.basename(filename))[0]
    export_csv = os.path.join(FLAGS.predict_dir, "beam_width_" + str(FLAGS.beam_width), str(video_id) + ".csv")
    logging.info("Working on {0}.".format(video_id))
    # Get data information
    data = tf.data.TFRecordDataset(filename)
    n = len(list(data))
    v_seq_length = n+seq_length-1
    # Get the aggregators
    labels_aggregator = aggregation.ConcatAggregator(n=n, idx=seq_length-1)
    logits_aggregator = aggregation.AverageAggregator(num_classes=num_classes, seq_length=seq_length)
    decode_fn = rep.get_decode_fn(1)
    preds_aggregator = aggregation.BatchLevelVotedPredsAggregator(
      num_classes=num_classes, seq_length=seq_length, def_val=DEF_VAL,
      decode_fn=decode_fn)
    # Iterate through batches
    for i, batch_data in enumerate(data):
      b_logits, b_labels = parse(batch_data)
      # Aggregation step
      labels_aggregator.step(i, b_labels)
      logits_aggregator.step(i, b_logits)
      preds_aggregator.step(i, b_logits)
    # Get aggregated data
    labels = labels_aggregator.result()
    logits = logits_aggregator.result()
    preds = preds_aggregator.result()
    # Collapse on video level
    preds = rep.get_inference_collapse_fn(v_seq_length)(preds)
    # Remove empty batch dimensions
    labels = tf.squeeze(labels, axis=0)
    logits = tf.squeeze(logits, axis=0)
    preds = tf.squeeze(preds, axis=0)
    # Update metrics for single stage model
    metrics.update(labels, preds)
    # Save
    ids = [video_id] * v_seq_length
    logging.info("Writing {0} examples to {1}.csv...".format(len(ids), video_id))
    save_array = np.column_stack((ids, labels.numpy().tolist(),
      logits.numpy().tolist(), preds.numpy().tolist()))
    np.savetxt(export_csv, save_array, delimiter=",", fmt='%s')
  # Print metrics
  metrics.finish()

# Run
if __name__ == "__main__":
  app.run(main=main)
