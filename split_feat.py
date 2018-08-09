"""Subsample the features so that there is no overlap."""

import h5py
import numpy as np
import os

f = h5py.File('data/sub_activitynet_v1-3.c3d.hdf5', 'r')

if not os.path.exists('data/feat'):
  os.mkdir('data/feat')

for i in f.keys():
  feat = f[i + '/c3d_features'][:]
  feat = feat[::2].astype(np.float32)
  np.save('data/feat/' + i, feat)
