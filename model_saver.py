"""Save checkpoints"""

from absl import logging
import glob
import os

class Checkpoint(object):
    """One checkpoint"""
    score = None
    path = None
    dir = None
    file = None

    def __init__(self, score, dir, file):
        self.score = score
        self.path = os.path.join(dir, file)
        self.dir = dir
        self.file = file

class ModelSaver(object):
    """Save the best model checkpoints to disk"""
    checkpoints = None
    dir = None
    metric = None
    keep_num = None
    compare_fn = None
    sort_reverse = None

    def __init__(self,
                 dir="checkpoints",
                 file="model_",
                 keep_num=5,
                 save_weights_only=True,
                 compare_fn=lambda x,y: x.score < y.score,
                 sort_reverse=False):
        """Init the ModelSaver"""
        self.checkpoints = []
        self.dir = dir
        self.keep_num = keep_num
        self.save_weights_only = save_weights_only
        self.compare_fn = compare_fn
        self.sort_reverse = sort_reverse

    def save(self, model, score, step, file):
        logging.info('Saving checkpoint for step %d' % (step,))
        file = file + "_" + str(step)
        checkpoint = Checkpoint(score, dir=self.dir, file=file)

        if len(self.checkpoints) < self.keep_num \
            or self.compare_fn(checkpoint, self.checkpoints[-1]):
            # Keep checkpoint
            logging.info("Keeping checkpoint {} with score {}".format(
                checkpoint.file, checkpoint.score))
            self.checkpoints.append(checkpoint)
            self.checkpoints = sorted(
                self.checkpoints, key=lambda x: x.score, reverse=self.sort_reverse)
            # The destination directory (make if necessary)
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            # Save model
            if self.save_weights_only:
                model.save_weights(checkpoint.path, overwrite=True)
            else:
                model.save(checkpoint.path, overwrite=True)
            # Prune checkpoints
            for checkpoint in self.checkpoints[self.keep_num:]:
                logging.info('Removing old checkpoint {} with score {}'.format(
                    checkpoint.file, checkpoint.score))
                for file in glob.glob(r'{}*'.format(checkpoint.path)):
                    os.remove(file)
            self.checkpoints = self.checkpoints[0:self.keep_num]

        else:
            # Skip the checkpoint
            logging.info('Skipping checkpoint {}'.format(checkpoint.file))
