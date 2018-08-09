"""Generates the dataset json files.

As described in https://arxiv.org/abs/1808.01575, we split the original videos
so that there is only one action segment in each video.
"""

from absl import app
from absl import flags
import json
import numpy as np

np.random.seed(0)

flags.DEFINE_integer('max_length', 300, 'max video length.')

FLAGS = flags.FLAGS


def start_location(n_frames, duration, start_time):
  """Computes the start location of a video segment.

  Args:
    n_frames: The number of frames in the original video.
    duration: The duration of the original video.
    start_time: The start time of a segment.

  Returns:
    location: The start location in the feature list of the original video.
  """
  return int((n_frames * start_time / duration) // 16)


def end_location(n_frames, n_feats, duration, end_time):
  """Computes the start location of a video segment.

  Args:
    n_frames: The number of frames in the original video.
    n_feats: The number of extracted from the original video.
    duration: The duration of the original video.
    end_time: The end time of a segment.

  Returns:
    location: The end location in the feature list of the original video.
  """
  location = int((n_frames * end_time / duration) // 16)
  location = min(n_feats, location + 1)
  return location


def get_videos(subset):
  """Splits the original videos."""
  with open('data/activity_net.v1-3.min.json', 'r') as f:
    info = json.load(f)
  with open('data/length.json', 'r') as f:
    length = json.load(f)

  # Gets the class names
  names = {}
  classes = {}
  for i in info['taxonomy']:
    classes[i['nodeName']] = i['nodeId']
    names[i['nodeId']] = i['nodeName']
  for i in info['taxonomy']:
    if i['parentId'] is not None and names[i['parentId']] in classes:
      del classes[names[i['parentId']]]
  assert len(classes) == 200

  perm = np.load('data/perm.npy')
  if subset == 'train':
    used = list(perm[:160])
  elif subset == 'val':
    used = list(perm[160:180])
  elif subset == 'test':
    used = list(perm[180:])
  else:
    raise ValueError('Unknown subset.')
  names = classes.keys()
  names.sort()
  names = [v for i, v in enumerate(names) if i in used]

  all_videos = []
  for k, v in info['database'].items():
    annos = v['annotations']
    annos = sorted(annos, key=lambda x: x['segment'][0])
    keep = True
    removed = []
    for i in range(len(annos)):
      for j in range(i + 1, len(annos)):
        if annos[j]['segment'][0] > annos[i]['segment'][1]:
          break
        # Overlapping segments.
        if annos[i]['label'] == annos[j]['label']:
          removed.append(j)
          if annos[i]['segment'][1] < annos[j]['segment'][1]:
            annos[i]['segment'][1] = annos[j]['segment'][1]
        else:
          keep = False
          break
      if not keep:
        break
    if not keep:
      continue
    removed = np.unique(removed)
    for i in removed[::-1]:
      annos.pop(i)

    for i in range(len(annos)):
      if annos[i]['label'] not in names:
        continue
      segment = annos[i]['segment']
      gt_start = start_location(length[k]['n_frames'], v['duration'],
                                segment[0])
      gt_end = end_location(length[k]['n_frames'], length[k]['n_feat'],
                            v['duration'], segment[1])
      # In the first version of the dataset, we limit the length of the
      # groundtruth segment.
      if gt_end - gt_start > 32 or gt_start >= gt_end:
        continue

      # Adds background segments before and after the action segment.
      ind_s = i
      while ind_s > 0 and annos[ind_s - 1]['label'] != annos[i]['label']:
        ind_s -= 1
      ind_e = i
      while ind_e + 1 < len(annos) and annos[ind_e + 1]['label'] != annos[i][
        'label']:
        ind_e += 1
      seg_start = start_location(length[k]['n_frames'], v['duration'],
                                 0 if ind_s == 0 else
                                 annos[ind_s - 1]['segment'][1])
      seg_end = end_location(length[k]['n_frames'], length[k]['n_feat'],
                             v['duration'],
                             v['duration'] if ind_e + 1 == len(annos) else
                             annos[ind_e + 1]['segment'][0])
      assert gt_start >= seg_start
      assert gt_end <= seg_end

      # We also limit the length of the whole video.
      if seg_end - seg_start > FLAGS.max_length:
        st = np.random.randint(max(gt_end - FLAGS.max_length, seg_start),
                               min(gt_start, seg_end - FLAGS.max_length) + 1)
        seg_start = st
        seg_end = st + FLAGS.max_length
      gt_start -= seg_start
      gt_end -= seg_start

      video = dict()
      video['id'] = k
      video['groundtruth'] = [gt_start, gt_end]
      video['label'] = classes[annos[i]['label']]
      video['location'] = [seg_start, seg_end]
      all_videos.append(video)
  return all_videos


def main(_):
  for i in ['train', 'val', 'test']:
    videos = get_videos(i)
    with open('data/%s.json' % i, 'w') as f:
      json.dump(videos, f)


if __name__ == '__main__':
  app.run(main)
