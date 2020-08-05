"""Save checkpoints"""

from absl import logging
import glob
import os

class Checkpoint(object):
  """A checkpoint with a score"""
  def __init__(self, score, dir, file):
    self.score = score
    self.path = os.path.join(dir, file)
    self.dir = dir
    self.file = file

class ModelSaver(object):
  """Save the best model checkpoints to disk"""
  def __init__(self,
         dir="checkpoints",
         keep_best=5,
         keep_latest=1,
         save_weights_only=True,
         compare_fn=lambda x,y: x.score < y.score,
         sort_reverse=False):
    """Init the ModelSaver"""
    self.best_checkpoints = []
    self.latest_checkpoints = []
    self.dir = dir
    self.keep_best = keep_best
    self.keep_latest = keep_latest
    self.save_weights_only = save_weights_only
    self.compare_fn_best = compare_fn
    self.compare_fn_latest = lambda x,y: x.score > y.score
    self.sort_reverse = sort_reverse

  def __save(self, model, checkpoint):
    # The destination directory (make if necessary)
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    # Save model
    if self.save_weights_only:
      model.save_weights(checkpoint.path, overwrite=True)
    else:
      model.save(checkpoint.path, overwrite=True)

  def save_keep(self, model, step, file):
    logging.info("Saving and keeping checkpoint for step {}".format(step))
    file = file + "_keep_" + str(step)
    checkpoint = Checkpoint(step, dir=self.dir, file=file)
    self.__save(model, checkpoint)

  def save_latest(self, model, step, file):
    logging.info("Saving latest checkpoint for step {}".format(step))
    file = file + "_latest_" + str(step)
    checkpoint = Checkpoint(step, dir=self.dir, file=file)
    if len(self.latest_checkpoints) < self.keep_latest \
      or self.compare_fn_latest(checkpoint, self.latest_checkpoints[-1]):
      # Keep checkpoint
      self.latest_checkpoints.append(checkpoint)
      self.latest_checkpoints = sorted(
        self.latest_checkpoints, key=lambda x: x.score, reverse=True)
      # Save checkpoint
      self.__save(model, checkpoint)
      # Prune checkpoints
      for checkpoint in self.latest_checkpoints[self.keep_latest:]:
        for file in glob.glob(r'{}*'.format(checkpoint.path)):
          os.remove(file)
      self.latest_checkpoints = self.latest_checkpoints[0:self.keep_latest]
    else:
      # Skip the checkpoint
      logging.info('Skipping checkpoint {}'.format(checkpoint.file))

  def save_best(self, model, score, step, file):
    logging.info('Saving checkpoint for step %d' % (step,))
    file = file + "_best_" + str(step)
    checkpoint = Checkpoint(score, dir=self.dir, file=file)

    if len(self.best_checkpoints) < self.keep_best \
      or self.compare_fn_best(checkpoint, self.best_checkpoints[-1]):
      # Keep checkpoint
      logging.info("Keeping checkpoint {} with score {}".format(
        checkpoint.file, checkpoint.score))
      self.best_checkpoints.append(checkpoint)
      self.best_checkpoints = sorted(
        self.best_checkpoints, key=lambda x: x.score, reverse=self.sort_reverse)
      # Save checkpoint
      self.__save(model, checkpoint)
      # Prune checkpoints
      for checkpoint in self.best_checkpoints[self.keep_best:]:
        logging.info('Removing old checkpoint {} with score {}'.format(
          checkpoint.file, checkpoint.score))
        for file in glob.glob(r'{}*'.format(checkpoint.path)):
          os.remove(file)
      self.best_checkpoints = self.best_checkpoints[0:self.keep_best]

    else:
      # Skip the checkpoint
      logging.info('Skipping checkpoint {}'.format(checkpoint.file))
