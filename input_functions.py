import json
import numpy as np
import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS


def sample(query_id, query_gt, query_label, query_loc, all_ids, all_gts,
           all_labels, all_locs, is_training):
  """For each query video, random sample a reference video.
  
  Args:
    query_id: Query video id.
    query_gt: The action segment in the query video.
    query_label: The action category of the query video.
    query_loc: The location of the video segment in the original whole video.
    all_ids: All video ids.
    all_gts: The action segments in all the videos.
    all_labels: The action categories of all the videos.
    all_locs: The locations of the all videos in their corresponding original
      whole video.
    is_training: Whether in training mode.
  Returns:
    query_id: Query video id.
    query_loc: The action segment in the original whole video.
    chosen_id: The reference video id.
    chosen_gt: The action segment in the reference video.
    chosen_loc: The location of the reference video in the original whole video.
  """
  same = tf.equal(all_labels, query_label)
  longer = tf.less_equal(query_gt[1] - query_gt[0],
                         all_locs[:, 1] - all_locs[:, 0])
  same = tf.logical_and(same, longer)
  same = tf.where(same)
  num = tf.shape(same)[0]
  idx = tf.random_uniform([], maxval=num, dtype=tf.int32,
                          seed=None if is_training else 6)
  idx = same[idx, 0]
  chosen_id = all_ids[idx]
  chosen_gt = all_gts[idx]
  chosen_loc = all_locs[idx]

  # Data augmentation during training.
  if is_training:
    off_st = tf.random_uniform([], maxval=chosen_gt[0] + 1, dtype=tf.int32)
    maxval = chosen_loc[1] - chosen_loc[0] - chosen_gt[1] + 1
    off_en = tf.random_uniform([], maxval=maxval, dtype=tf.int32)
    use_off = tf.random_uniform([])
    off_st = tf.cond(use_off < 0.9, lambda: off_st, lambda: 0)
    off_en = tf.cond(use_off < 0.9, lambda: off_en, lambda: 0)
    off_gt = tf.stack([-off_st, -off_st])
    off_loc = tf.stack([off_st, -off_en])
    chosen_gt += off_gt
    chosen_loc += off_loc
  return query_id, query_gt + query_loc[0], chosen_id, chosen_gt, chosen_loc


def batching_func(x, batch_size):
  return x.padded_batch(
    batch_size,
    padded_shapes=(
      tf.TensorShape([None, FLAGS.feat_dim]),
      tf.TensorShape([]),
      tf.TensorShape([None, FLAGS.feat_dim]),
      tf.TensorShape([]),
      tf.TensorShape([2])))


def input_fn(subset, batch_size):
  is_training = subset == 'train'
  with open(os.path.join('data', subset + '.json'), 'r') as f:
    data = json.load(f)
  videos = [[] for _ in range(4)]
  for i in data:
    videos[0].append(i['id'])
    videos[1].append(i['groundtruth'])
    videos[2].append(i['label'])
    videos[3].append(i['location'])
  for i in range(4):
    videos[i] = tf.convert_to_tensor(videos[i])
  dataset = tf.data.Dataset.from_tensor_slices(tuple(videos))
  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
  dataset = dataset.map(
    lambda v, t, l, d: sample(v, t, l, d, *videos, is_training=is_training))
  dataset = dataset.map(
    lambda v1, t1, v2, t2, l2: tuple(
      tf.py_func(get_data, [FLAGS.data_dir, v1, t1, v2, t2, l2],
                 [tf.float32, tf.int32, tf.float32, tf.int32, tf.int32])))

  if is_training:

    def key_func(unused_1, len1, unused_2, len2, unused_3):
      id2 = len2 // FLAGS.bucket_span
      return tf.to_int64(id2)

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data, batch_size)

    batched_dataset = dataset.apply(
      tf.contrib.data.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = batching_func(dataset, batch_size)

  dataset = batched_dataset.map(reorder_func)
  dataset = dataset.prefetch(4)
  return dataset


def get_data(data_dir, v1, t1, v2, t2, l2):
  """Read the video features."""
  feat1 = np.load('%s/feat/v_%s.npy' % (data_dir, v1))
  feat2 = np.load('%s/feat/v_%s.npy' % (data_dir, v2))
  len1 = t1[1] - t1[0]
  len2 = l2[1] - l2[0]
  ret1 = feat1[t1[0]:t1[1]]
  ret2 = feat2[l2[0]:l2[1]]
  assert len1 == ret1.shape[0]
  assert len2 == ret2.shape[0]
  assert np.all(t2 >= 0) and np.all(t2 <= len2)
  return ret1, len1, ret2, len2, t2


def reorder_func(v1, l1, v2, l2, label):
  # v1.set_shape([FLAGS.batch_size, None, FLAGS.feat_dim])
  # l1.set_shape([FLAGS.batch_size])
  # v2.set_shape([FLAGS.batch_size, None, FLAGS.feat_dim])
  # l2.set_shape([FLAGS.batch_size])
  # label.set_shape([FLAGS.batch_size, 2])
  return (v1, l1, v2, l2), label
