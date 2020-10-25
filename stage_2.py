"""Evaluate exported frame-level probabilities."""

from __future__ import division
import argparse
import csv
import glob
import numpy as np
import os
from scipy.special import softmax
import sys

CSV_SUFFIX = '*.csv'

np.set_printoptions(threshold=sys.maxsize)

def import_probs_and_labels(args):
  """Import probabilities and labels from csv"""
  filenames = sorted(glob.glob(os.path.join(args.input_dir, CSV_SUFFIX)))
  assert filenames, "No files found for evaluation"
  labels = {}
  probs = {}
  for filename in filenames:
    labels[filename] = []
    probs[filename] = []
    with open(filename) as dest_f:
      for row in csv.reader(dest_f, delimiter=','):
        labels[filename].append(int(float(row[args.col_label])))
        all_inputs_t = []
        for i in range(args.num_event_classes + 1):
          all_inputs_t.append(float(row[args.col_input + i]))
        if args.input_format == "probs":
          probs[filename].append(np.array(all_inputs_t)[1:].tolist())
        elif args.input_format == "logits":
          probs[filename].append(softmax(all_inputs_t)[1:].tolist())
    labels[filename] = np.array(labels[filename])
    probs[filename] = np.array(probs[filename])
  return probs, labels

def max_search(probs, threshold, mindist, def_val):
  """Perform a max search"""
  # Threshold probs without default event probs
  probabilities = np.copy(probs)
  probabilities[probabilities <= threshold] = 0
  # Return array
  detections = np.empty(np.shape(probabilities)[0], dtype=np.int32)
  detections.fill(def_val)
  # Potential detections
  idx_p = np.where(probabilities > 0)[0]
  if idx_p.size == 0:
    return detections
  # Identify start and end of detections
  p_d = np.diff(idx_p) - 1
  p = np.where(p_d > 0)[0]
  p_start = np.concatenate(([0], p+1))
  p_end = np.concatenate((p, [idx_p.shape[0]-1]))
  # Infer start and end indices of detections
  idx_start = idx_p[p_start]
  idx_end = idx_p[p_end]
  idx_max = [max(min(start+np.argmax(probabilities[start:end+1]), end), start)
    for start, end in zip(idx_start, idx_end)]
  # Remove detections within mindist
  max_diff = np.diff(idx_max)
  carry = 0; rem_i = []
  for i, diff in enumerate(np.concatenate(([mindist], max_diff))):
    if (diff + carry < mindist):
      rem_i.append(i)
      carry += diff
    else:
      carry = 0
  if len(rem_i) > 0:
    idx_max_mindist = np.delete(idx_max, rem_i)
  else:
    idx_max_mindist = idx_max
  # Return detections
  detections[idx_max_mindist] = np.argmax(probabilities[idx_max_mindist], axis=-1) + def_val + 1
  return detections

def eval_stage_2(dets, labels, event_val, def_val):
  """Stage 2 evaluation based on gesture-level metric proposed by Kyritsis et al. (2019)"""
  def _split_idx(labels):
    idx_t = np.where(labels == event_val)[0]
    t_d = np.diff(idx_t) - 1
    t = np.where(t_d > 0)[0]
    t_start = np.concatenate(([0], t+1))
    t_end = np.concatenate((t, [idx_t.shape[0]-1]))
    if len(idx_t > 0):
      idx_start = idx_t[t_start]
      idx_end = idx_t[t_end]
    else:
      return []
    return [np.arange(start, end+1) for start, end in zip(idx_start, idx_end)]
  idxs_t = _split_idx(labels)
  idxs_f = np.where(labels == def_val)
  idxs_o = np.intersect1d(np.where(labels != def_val), np.where(labels != event_val))
  splits_t = [dets[split_idx] for split_idx in idxs_t]
  splits_f = dets[idxs_f]
  splits_o = dets[idxs_o]
  tp = np.sum([1 if np.sum(np.equal(split, event_val)) > 0 else 0 for split in splits_t])
  fn = np.sum([0 if np.sum(np.equal(split, event_val)) > 0 else 1 for split in splits_t])
  fp_1 = np.sum([np.sum(np.equal(split, event_val)) - 1 if np.sum(np.equal(split, event_val)) > 1 else 0 for split in splits_t])
  fp_2 = np.sum(np.equal(splits_f, event_val))
  fp_3 = np.sum(np.equal(splits_o, event_val))
  if tp > 0:
    prec = tp / (tp + fp_1 + fp_2 + fp_3)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
  elif fn == 0:
    prec = 1
    rec = 1
    f1 = 1
  else:
    prec = 0
    rec = 0
    f1 = 0
  return tp, fn, fp_1, fp_2, fp_3, prec, rec, f1

def main(args=None):
  # Event classes excluding default/idle
  event_classes = range(args.def_val + 1, args.def_val + args.num_event_classes + 1, 1)
  # Import the probs and labels from csv
  probs, labels = import_probs_and_labels(args)
  # Perform grid search
  if args.mode == 'estimate':
    # Collect all in one array
    flat_labels = np.array([label for f in labels.keys() for label in labels[f]])
    flat_probs = np.array([prob for f in probs.keys() for prob in probs[f]])
    # All evaluated threshold values
    threshold_vals = np.arange(args.min_threshold, args.max_threshold, args.inc_threshold)
    f1_results = []
    for threshold in threshold_vals:
      # Perform max search
      flat_dets = np.array([det for f in probs.keys() for det in
        max_search(probs[f], threshold, args.min_dist, args.def_val)])
      # Calculate Stage II
      f1 = []; pre = []; rec = []
      for i, event_val in enumerate(event_classes):
        _, _, _, _, _, _, _, f1_i = eval_stage_2(flat_dets, flat_labels, event_val, args.def_val)
        f1.append(f1_i)
      f1_results.append(np.mean(f1))
    # Find best threshold
    final_threshold = threshold_vals[np.argmax(f1_results)]
    print("===================================================")
    print('Best threshold: {}'.format(final_threshold))
    final_dets = max_search(flat_probs.tolist(), final_threshold, args.min_dist, args.def_val)
    f1 = []; pre = []; rec = []
    for i, event_val in enumerate(event_classes):
      tp_i, fn_i, fp_1_i, fp_2_i, fp_3_i, pre_i, rec_i, f1_i = eval_stage_2(
        final_dets, flat_labels, event_val, args.def_val)
      f1.append(f1_i); pre.append(pre_i); rec.append(rec_i)
      # Print results
      print('---------------------- Class {} --------------------'.format(event_val))
      print('F1: {}'.format(f1_i))
      print('Precision: {}'.format(pre_i))
      print('Recall: {}'.format(rec_i))
      print('-----')
      print('TP: {}'.format(tp_i))
      print('FP_1: {}'.format(fp_1_i))
      print('FP_2: {}'.format(fp_2_i))
      print('FP_3: {}'.format(fp_3_i))
      print('FN: {}'.format(fn_i))
    print("===================================================")
    print('mF1: {}'.format(np.mean(f1)))
    print('mPre: {}'.format(np.mean(pre)))
    print('mRec: {}'.format(np.mean(rec)))
  else:
    # Perform max search
    tp, fp_1, fp_2, fp_3, fn = {}, {}, {}, {}, {}
    for e in event_classes:
      tp[str(e)], fp_1[str(e)], fp_2[str(e)], fp_3[str(e)], fn[str(e)] = \
        [], [], [], [], []
    for f in probs.keys():
      print('---------------------- ID {} --------------------'.format(f))
      # Max search for f
      dets_f = max_search(probs[f], args.threshold, args.min_dist, args.def_val)
      # Calculate Stage II
      for i, e in enumerate(event_classes):
        tp_i, fn_i, fp_1_i, fp_2_i, fp_3_i, pre_i, rec_i, f1_i = eval_stage_2(
          dets_f, labels[f], e, args.def_val)
        tp[str(e)].append(tp_i); fp_1[str(e)].append(fp_1_i);
        fp_2[str(e)].append(fp_2_i); fp_3[str(e)].append(fp_3_i);
        fn[str(e)].append(fn_i)
        # Print results
        print('---------------------- Class {} --------------------'.format(e))
        print('F1: {}'.format(f1_i))
        print('Precision: {}'.format(pre_i))
        print('Recall: {}'.format(rec_i))
        print('-----')
        print('TP: {}'.format(tp_i))
        print('FP_1: {}'.format(fp_1_i))
        print('FP_2: {}'.format(fp_2_i))
        print('FP_3: {}'.format(fp_3_i))
        print('FN: {}'.format(fn_i))
    print("===================================================")
    f1s, pres, recs = [], [], []
    for e in event_classes:
      print('---------------------- Class {} --------------------'.format(e))
      tp_e = np.sum(tp[str(e)])
      fp_1_e = np.sum(fp_1[str(e)])
      fp_2_e = np.sum(fp_2[str(e)])
      fp_3_e = np.sum(fp_3[str(e)])
      fn_e = np.sum(fn[str(e)])
      if tp_e > 0:
        pre_e = tp_e / (tp_e + fp_1_e + fp_2_e + fp_3_e)
        rec_e = tp_e / (tp_e + fn_e)
        f1_e = 2 * pre_e * rec_e / (pre_e + rec_e)
      elif fn_e == 0:
        pre_e = 1
        rec_e = 1
        f1_e = 1
      else:
        pre_e = 0
        rec_e = 0
        f1_e = 0
      pres.append(pre_e)
      recs.append(rec_e)
      f1s.append(f1_e)
      print('F1: {}'.format(f1_e))
      print('Precision: {}'.format(pre_e))
      print('Recall: {}'.format(rec_e))
      print('-----')
      print('TP: {}'.format(tp_e))
      print('FP_1: {}'.format(fp_1_e))
      print('FP_2: {}'.format(fp_2_e))
      print('FP_3: {}'.format(fp_3_e))
      print('FN: {}'.format(fn_e))
    print('mF1: {}'.format(np.mean(f1s)))
    print('mPre: {}'.format(np.mean(pres)))
    print('mRec: {}'.format(np.mean(recs)))

# Run
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate model Stage II')
  parser.add_argument('--input_dir', type=str, default='eval', nargs='?', help='Directory with eval data.')
  parser.add_argument('--min_dist', type=int, default=16, nargs='?', help='Minimum frames between detections.')
  parser.add_argument('--threshold', type=float, default=0.9, nargs='?', help='Detection threshold probability')
  parser.add_argument('--mode', type=str, default='evaluate', nargs='?', help='Evaluation or estimation and evaluation')
  parser.add_argument('--min_threshold', type=float, default=0.5, nargs='?', help='Minimum detection threshold probability')
  parser.add_argument('--max_threshold', type=float, default=1, nargs='?', help='Maximum detection threshold probability')
  parser.add_argument('--inc_threshold', type=float, default=0.001, nargs='?', help='Increment for detection threshold search')
  parser.add_argument('--col_label', type=int, default=1, nargs='?', help='Col number of label in csv')
  parser.add_argument('--col_input', type=int, default=2, nargs='?', help='First col number of event class logits or probs input in csv')
  parser.add_argument('--num_event_classes', type=int, default=1, nargs='?', help='Number of event classes excluding default/idle')
  parser.add_argument('--def_val', type=int, default=1, nargs='?', help='Value denoting default/idle event')
  parser.add_argument('--input_format', type=str, default='probs', choices=('probs', 'logits'), help='Format of the input class values')
  args = parser.parse_args()
  main(args)
