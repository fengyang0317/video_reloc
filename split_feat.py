"""Subsample the features so that there is no overlap."""

from absl import app
from absl import flags

import h5py
import numpy as np
import os

flags.DEFINE_string('feat_dir', 'data/feat/', 'feature directory')

FLAGS = flags.FLAGS


def main(_):
  f = h5py.File('data/sub_activitynet_v1-3.c3d.hdf5', 'r')

  if not os.path.exists(FLAGS.feat_dir):
    os.mkdir(FLAGS.feat_dir)

  for i in f.keys():
    feat = f[i + '/c3d_features'][:]
    feat = feat[::2].astype(np.float32)
    np.save(os.path.join(FLAGS.feat_dir, i), feat)


if __name__ == '__main__':
  app.run(main)
