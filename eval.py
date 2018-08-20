from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf

from input_functions import input_fn
from match import model_fn
from utils import get_distribution_strategy

tf.flags.DEFINE_string('eval_dir', 'eval',
                       'Directory where to write event logs.')

tf.flags.DEFINE_string('subset', 'test', 'subset')

tf.flags.DEFINE_string('ckpt_path', None, 'checkpoint path')

FLAGS = tf.flags.FLAGS


def main(_):
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)

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

  eval_input_fn = functools.partial(
    input_fn,
    subset=FLAGS.subset,
    batch_size=FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.eval_dir,
    config=run_config,
    params=FLAGS)

  estimator.evaluate(input_fn=eval_input_fn, checkpoint_path=FLAGS.ckpt_path)


if __name__ == '__main__':
  tf.app.run()
