"""Implementation of the method in Dong et al. (2012) for Clemson dataset."""

import os
import numpy as np
from absl import app
from absl import flags
from absl import logging
import itertools
import tensorflow as tf
from tensorflow.python.platform import gfile
from metrics import evaluate_interval_detection
import clemson
import oreba_dis
import fic

FLAGS = flags.FLAGS
flags.DEFINE_enum(name='dataset',
    default='oreba-dis', enum_values=["oreba-dis", "fic", "clemson"],
    help='Select the dataset')
flags.DEFINE_string(name='data_dir',
    default='data/Clemson_no_std/train', help='Directory to search for data.')
flags.DEFINE_enum(name='mode',
    default="evaluate", enum_values=["estimate", "evaluate"],
    help='Estimate parameters or evaluate with existing parameters')
flags.DEFINE_integer(name='T1',
    default=10, help='Parameter T1.')
flags.DEFINE_integer(name='T2',
    default=10, help='Parameter T2.')
flags.DEFINE_integer(name='T3',
    default=2, help='Parameter T3.')
flags.DEFINE_integer(name='T4',
    default=6, help='Parameter T4.')


def evaluate_with_parameters(dataset, filenames, T1, T2, T3, T4, freq):
    # Initialize metrics
    tp = 0; fp_1 = 0; fp_2 = 0; fn = 0
    # Loop through the files
    for filename in filenames:
        logging.info("Working on {0}.".format(filename))
        # Get the dataset. Use predicting mode without shuffling or batching.
        data = dataset(batch_size=1, is_training=False,
            is_predicting=True, data_dir=filename, num_shuffle=1)

        # From paper:
        # Let EVENT = 0
        # Loop
        #   Let Vt = measured roll vel. at time t
        #   If Vt > T1 and EVENT = 0
        #     EVENT = 1
        #     Let s = t
        #   if Vt < T2 and t-s > T3 and EVENT = 1
        #     Bite detected
        #     Let s = t
        #     EVENT = 2
        #   if EVENT = 2 and t-s > T4
        #     EVENT = 0

        preds = []
        labels = []

        logging.info("Running algorithm")

        s = 0
        event = 0
        # Loop through time frames
        for t, (features, label) in enumerate(data):
            labels.append(int(label))
            # Roll velocity is the radial velocity of x axis in features
            if FLAGS.dataset == 'clemson':
                v_t = features[:,:,5]
            elif FLAGS.dataset == 'fic' or FLAGS.dataset == 'oreba-dis':
                v_t = features[:,:,4]
            #logging.info("Vt = {0} label = {1}".format(v_t, int(label)))
            # Wrist roll velocity must surpass a positive threshold.
            if int(v_t) > T1 and event == 0:
                event = 1
                s = t
            # A minimum amount of time must pass and
            #  the velocity must surpass a negative threshold.
            # T3 and T4 are in seconds, so multiply by 15 Hz
            if int(v_t) < T2 and t-s > T3 * freq and event == 1:
                s = t
                event = 2
                preds.append(1)
            else:
                preds.append(0)
            if event == 2 and t-s > T4 * freq:
                event = 0

        assert len(labels) == len(preds)

        logging.info("Evaluating")

        tp_, fp_1_, fp_2_, _, fn_ = evaluate_interval_detection(
            labels=tf.expand_dims(labels, 0),
            predictions=tf.expand_dims(preds, 0),
            event_val=1, def_val=0, seq_length=len(preds))

        tp += tp_
        fp_1 += fp_1_
        fp_2 += fp_2_
        fn += fn_

    pre = tp / (tp + fp_1 + fp_2)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)

    return tp, fp_1, fp_2, fn, f1


def estimate_parameters(dataset, filenames, freq):
    """Estimate paramter values from a range of predetermined values"""
    # Param options
    param_options = [(T12, -T12, T3, T4) for T12, T3, T4 in list(itertools.product( \
        range(25, 30, 5), range(1, 4, 1), range(2, 10, 2)))]
    # Evaluate options
    all_f1 = []
    for T1, T2, T3, T4 in param_options:
        logging.info("Evaluating params T1={0}, T2={1}, T3={2}, T4={3}".format(
            T1, T2, T3, T4))
        _, _, _, _, f1 = evaluate_with_parameters(
            dataset=dataset, filenames=filenames, T1=T1, T2=T2, T3=T3, T4=T4,
            freq=freq)
        all_f1.append(f1)

    return param_options[np.argmax(all_f1)], np.max(all_f1)


def main(arg=None):
    # Read data frame-by-frame
    # Get dataset
    if FLAGS.dataset == 'oreba-dis':
        freq = 64
        dataset = oreba_dis.Dataset(label_mode="label_1", input_mode="inert",
            input_length=1, seq_shift=1)
    elif FLAGS.dataset == 'fic':
        freq = 100
        dataset = fic.Dataset(label_mode="label_1", input_length=1,
            seq_shift=1)
    elif FLAGS.dataset == 'clemson':
        freq = 15
        dataset = clemson.Dataset(label_mode="label_1", input_length=1,
            seq_shift=1)
    else:
        raise ValueError("Dataset {} not implemented!".format(FLAGS.dataset))
    filenames = gfile.Glob(os.path.join(FLAGS.data_dir, "*.tfrecord"))

    if FLAGS.mode == "evaluate":
        tp, fp_1, fp_2, fn, f1 = evaluate_with_parameters(
            dataset=dataset, filenames=filenames,
            T1=FLAGS.T1, T2=FLAGS.T2, T3=FLAGS.T3, T4=FLAGS.T4, freq=freq)
        logging.info("tp={}, fp_1={}, fp_2={}, fn={}".format(
            tp, fp_1, fp_2, fn))
        logging.info("F1 = {}".format(f1))
    elif FLAGS.mode == "estimate":
        params, f1 = estimate_parameters(dataset=dataset, filenames=filenames,
            freq=freq)
        logging.info("Best parameters: T1={0}, T2={1}, T3={2}, T4={3}".format(
            params[0], params[1], params[2], params[3]))
        logging.info("F1 = {}".format(f1))

# Run
if __name__ == "__main__":
    app.run(main=main)
