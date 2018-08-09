"""Gets the number of frames and features of each video."""

import cv2
import glob
import json
import numpy as np
import os

with open('data/activity_net.v1-3.min.json', 'r') as f:
  info = json.load(f)

all_files = {}
for i in glob.glob('data/videos/v_*'):
  name = os.path.split(i)[1]
  name = os.path.splitext(name)[0]
  all_files[name[2:]] = i

data = {}
for k, v in info['database'].items():
  feat = np.load('data/feat/v_%s.npy' % k)
  cap = cv2.VideoCapture(all_files[k])
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  data[k] = {}
  data[k]['n_frames'] = n_frames
  n_feat = feat.shape[0]
  data[k]['n_feat'] = n_feat
  if n_frames // 16 != n_feat:
    print '%s %d %d\t%d' % (k, n_feat, n_frames // 16, n_feat - n_frames // 16)

with open('data/length.json', 'w') as f:
  json.dump(data, f)
