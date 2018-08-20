import numpy as np
import tensorflow as tf


def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.
  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.
  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
        num_gpus=num_gpus,
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
          all_reduce_alg, num_packs=num_gpus))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def get_eval_metric(iou):
  th = np.arange(0.1, 1.0, 0.1)
  vals = [tf.to_float(iou > i) for i in th]
  metrics = dict([('IoU/%g' % i, tf.metrics.mean(j)) for i, j in zip(th, vals)])
  metrics['IoU/mean'] = tf.metrics.mean(vals[4:])
  return metrics


def get_iou(pred, label):
  pred_l, pred_r = tf.unstack(pred, axis=1)
  for i in range(2, len(pred.shape)):
    label = tf.expand_dims(label, axis=i)
  label_l, label_r = tf.unstack(label, axis=1)
  inter_l = tf.maximum(pred_l, label_l)
  inter_r = tf.minimum(pred_r, label_r)
  inter = tf.maximum(inter_r - inter_l, 0)
  union = pred_r - pred_l + label_r - label_l - inter
  return tf.divide(inter, union, name='iou')
