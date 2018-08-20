from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_functions import input_fn
from match_cell_wrapper import MatchCellWrapper
from utils import get_distribution_strategy
from utils import get_eval_metric
from utils import get_iou

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('max_length', 300, 'max length')

tf.flags.DEFINE_integer('feat_dim', 500, 'feature dim')

tf.flags.DEFINE_float('keep_prob', 0.6, 'keep prob')

tf.flags.DEFINE_integer('mem_dim', 128, 'hidden state dim')

tf.flags.DEFINE_integer('att_dim', 128, 'attention dim')

tf.flags.DEFINE_string('job_dir', 'saving', 'job dir')

tf.flags.DEFINE_string('data_dir', '/home/yfeng23/dataset/activity_net/',
                       'data dir')

tf.flags.DEFINE_integer('num_gpus', 0, 'number gpus')

tf.flags.DEFINE_integer('bucket_span', 30, 'bucket span')

tf.flags.DEFINE_integer('batch_size', 128, 'batch size')

tf.flags.DEFINE_integer('max_steps', 1000, 'training steps')

tf.flags.DEFINE_float('weight_decay', 0.0, 'weight decay')

tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')

tf.flags.DEFINE_float('max_gradient_norm', 5.0, 'max gradient norm')

tf.flags.DEFINE_integer('save_summary_steps', 10, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 100, 'save ckpt')

FLAGS = tf.flags.FLAGS


def model_fn(features, labels, mode, params):
  """Model function."""

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  query, len_q, ref, len_r = features
  batch_size = tf.shape(query)[0]

  # Video feature aggregation (Sec. 3.1).
  cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim)
  with tf.variable_scope('video_lstm', reuse=tf.AUTO_REUSE):
    out1, state1 = tf.nn.dynamic_rnn(cell, query, len_q, dtype=tf.float32)
    out2, state2 = tf.nn.dynamic_rnn(cell, ref, len_r, dtype=tf.float32)
  out1 = slim.dropout(out1, keep_prob=params.keep_prob, is_training=is_training)
  out2 = slim.dropout(out2, keep_prob=params.keep_prob, is_training=is_training)

  # Matching (Sec. 3.2).
  forward = tf.nn.rnn_cell.BasicLSTMCell(params.att_dim, name='forward')
  forward = MatchCellWrapper(forward, out1, len_q)
  backward = tf.nn.rnn_cell.BasicLSTMCell(params.att_dim, name='backward')
  backward = MatchCellWrapper(backward, out1, len_q, reuse=tf.AUTO_REUSE)
  with tf.variable_scope('att'):
    forward_out, forward_state = tf.nn.dynamic_rnn(forward, out2, len_r,
                                                   dtype=tf.float32)
    out2_reverse = tf.reverse_sequence(out2, len_r, 1, 0)
    backward_out, backward_state = tf.nn.dynamic_rnn(backward, out2_reverse,
                                                     len_r, dtype=tf.float32)
    backward_out = tf.reverse_sequence(backward_out, len_r, 1, 0)
  h = tf.concat([forward_out, backward_out], axis=2, name='concat_H')
  h = slim.dropout(h, keep_prob=params.keep_prob + 0.2, is_training=is_training)

  # Localization (Section 3.3).
  pointer = tf.nn.rnn_cell.BasicLSTMCell(params.att_dim)
  maxlen = tf.shape(h)[1]
  with tf.variable_scope('pointer'):
    point_out, _ = tf.nn.dynamic_rnn(pointer, h, len_r, dtype=tf.float32)
    logits = slim.fully_connected(point_out, 4, activation_fn=None, scope='loc')

  # Make predictions.
  def map_body(x):
    logits = x[0]
    length = x[1]
    logits = logits[:length]
    prob = tf.nn.log_softmax(logits, axis=1)
    prob = tf.transpose(prob)

    initial_it = tf.constant(0, dtype=tf.int32)
    initial_idx_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True,
                                    element_shape=[])
    initial_val_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                    element_shape=[])

    def cond(it, *unused):
      # Limits the length to be smaller than 1024 frames.
      return it < tf.minimum(length, 64)

    def while_body(it, idx_ta, val_ta):
      # Eq. (11) is implemented here.
      total = tf.cond(tf.equal(it, 0), lambda: tf.reduce_sum(prob[:2], axis=0),
                      lambda: prob[0, :-it] + prob[1, it:])

      def get_inside():
        score = tf.tile(prob[2, None, :], [it, 1])
        score = tf.reverse_sequence(score, tf.zeros([it], tf.int32) + length, 1,
                                    0)
        score = tf.reverse_sequence(score, length - tf.range(it), 1, 0)
        score = score[:, :-it]
        score = tf.reduce_mean(score, axis=0)
        return score

      ave = tf.cond(tf.equal(it, 0), lambda: prob[2], get_inside)
      total += ave
      idx = tf.argmax(total, output_type=tf.int32, name='max1')
      idx_ta = idx_ta.write(it, idx)
      val_ta = val_ta.write(it, total[idx])
      it += 1
      return it, idx_ta, val_ta

    res = tf.while_loop(cond, while_body,
                        [initial_it, initial_idx_ta, initial_val_ta])
    final_idx = res[1].stack()
    final_val = res[2].stack()
    idx = tf.argmax(final_val, output_type=tf.int32)
    pred = tf.stack([final_idx[idx], final_idx[idx] + idx + 1])
    return pred

  predictions = tf.map_fn(map_body, [logits, len_r], tf.int32)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
    )

  # Loss computation.
  idx = tf.stack([tf.range(batch_size), labels[:, 0]], axis=1)
  label_st = tf.scatter_nd(idx, tf.ones(batch_size), [batch_size, maxlen])
  idx = tf.stack([tf.range(batch_size), labels[:, 1] - 1], axis=1)
  label_en = tf.scatter_nd(idx, tf.ones(batch_size), [batch_size, maxlen])
  inside_t = tf.sequence_mask(labels[:, 1] - labels[:, 0], maxlen)
  inside_t = tf.reverse_sequence(inside_t, labels[:, 1], 1, 0)
  outside = tf.logical_not(inside_t)
  inside_t = tf.to_float(inside_t)
  outside = tf.to_float(outside)
  label = tf.stack([label_st, label_en, inside_t, outside], axis=2)

  # Eq. (10)
  heavy = tf.reduce_sum(label[:, :, :2], axis=-1) > 0.9
  heavy = tf.to_float(heavy) * 9 + 1
  label = label / tf.reduce_sum(label, axis=2, keepdims=True)
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
  loss *= heavy
  mask = tf.sequence_mask(len_r, maxlen)
  loss = tf.boolean_mask(loss, mask)
  loss = tf.reduce_mean(loss)
  model_params = tf.trainable_variables()
  weights = [i for i in model_params if 'bias' not in i.name]
  loss += params.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in weights])

  # Optimization.
  gradients = tf.gradients(loss, model_params)
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
    gradients, params.max_gradient_norm)
  tf.summary.scalar('grad_norm', gradient_norm)
  tf.summary.scalar('clipped_gradient', tf.global_norm(clipped_gradients))

  # boundaries = [200, 400, 600]
  # staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]
  # learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
  #                                             boundaries, staged_lr)
  # tf.summary.scalar('learning_rate', learning_rate)
  tensors_to_log = {'loss': loss, 'step': tf.train.get_global_step(),
                    'len_q': tf.shape(features[0])[1],
                    'len_r': tf.shape(features[2])[1]}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                            every_n_iter=10)
  train_hooks = [logging_hook]
  optimizer = tf.train.AdamOptimizer(params.learning_rate)

  if is_training:
    train_op = optimizer.apply_gradients(zip(clipped_gradients, model_params),
                                         tf.train.get_global_step())
  else:
    train_op = None

  # Evaluation.
  iou = get_iou(predictions, labels)
  metrics = get_eval_metric(iou)

  for variable in tf.trainable_variables():
    tf.summary.histogram(variable.op.name, variable)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    training_hooks=train_hooks,
    eval_metric_ops=metrics)


def main(_):
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    gpu_options=tf.GPUOptions(allow_growth=True))

  distribution_strategy = get_distribution_strategy(FLAGS.num_gpus)

  run_config = tf.estimator.RunConfig(
    session_config=session_config,
    save_checkpoints_steps=FLAGS.save_checkpoint_steps,
    save_summary_steps=FLAGS.save_summary_steps,
    keep_checkpoint_max=100,
    train_distribute=distribution_strategy)

  train_input_fn = functools.partial(
    input_fn,
    subset='train',
    batch_size=FLAGS.batch_size)

  eval_input_fn = functools.partial(
    input_fn,
    subset='val',
    batch_size=FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.job_dir,
    config=run_config,
    params=FLAGS)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=FLAGS.max_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                    throttle_secs=9)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.app.run()
