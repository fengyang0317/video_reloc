import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn_cell_impl import RNNCell


class MatchCellWrapper(RNNCell):
  """Match the query video feature with the reference video feature."""

  def __init__(self, cell, hq, length, reuse=None):
    """Initialize the cell.

    Args:
      cell: An RNNCell.
      hq: The reference video feature.
      length: The length of the reference video.
      reuse: Whether reuse the parameters.
    """
    super(MatchCellWrapper, self).__init__(_reuse=reuse)
    self._cell = cell
    self._hq = hq
    self._length = length
    self._attn_vec_size = cell.output_size

  @property
  def output_size(self):
    return self._attn_vec_size

  @property
  def state_size(self):
    return self._cell.state_size

  def call(self, inputs, state):
    """Matching operations described in Sec. 3.2 in the paper.

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state.
    """
    hq = slim.fully_connected(
      self._hq, self._attn_vec_size, activation_fn=None,
      biases_initializer=None, scope='fc_hq')
    c, h = state
    concat = tf.concat([inputs, h], axis=1)
    hp = slim.fully_connected(concat, self._attn_vec_size, activation_fn=None,
                              scope='fc_hp')
    hp = tf.expand_dims(hp, 1)
    g = tf.tanh(hq + hp, name='match_tanh')
    g = slim.fully_connected(g, 1, activation_fn=None, scope='fc_g')
    g = tf.squeeze(g, 2)

    def body(x):
      alpha = x[0]
      hq = x[1]
      length = x[2]
      alpha = alpha[:length]
      hq = hq[:length]
      alpha = tf.nn.softmax(alpha)
      hq = tf.reduce_sum(hq * alpha[:, None], axis=0)
      return hq

    hq = tf.map_fn(body, [g, self._hq, self._length], tf.float32)

    # Cross gating.
    gate_hq = slim.fully_connected(inputs, int(hq.shape[1]), tf.sigmoid,
                                   scope='gate_q')
    gate_in = slim.fully_connected(hq, int(inputs.shape[1]), tf.sigmoid,
                                   scope='gate_r')
    hq *= gate_hq
    inputs *= gate_in
    inputs = bilinear(inputs, hq)
    cell_output, new_state = self._cell(inputs, state)
    return cell_output, new_state


def bilinear(x, y, num_outputs=None, k=8):
  """Factorized bilinear matching.

  Args:
    x: The first vector.
    y: The second vector.
    num_outputs: The dimension of the output vector.
    k: The dimension of projected space.

  Returns: The matching results.
  """
  _, input_dim = x.get_shape().as_list()
  if num_outputs is None:
    num_outputs = input_dim
  with tf.variable_scope('bi', reuse=tf.AUTO_REUSE):
    x = slim.fully_connected(x, num_outputs * k, activation_fn=None, scope='fc')
    y = slim.fully_connected(y, num_outputs * k, activation_fn=None, scope='fc')
  x = tf.reshape(x, [-1, num_outputs, k])
  y = tf.reshape(y, [-1, num_outputs, k])
  bi = tf.reduce_sum(x * y, axis=-1)
  return bi
