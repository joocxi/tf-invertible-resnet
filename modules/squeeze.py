from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class Squeeze:
  """
  Squeezing layer for multi-scale block
  """
  def __init__(self,
               in_shape,
               factor=2):
    self.factor = factor
    self.in_shape = in_shape
    self.out_shape = self.get_outshape()

  def __call__(self, x):
    """

    :param x: (batch_size, height, width, channels)
    :return out: (batch_size, height/factor, width/factor, factor^2*channels)
    """

    (batch_size, height, width, channels) = self.in_shape
    out = tf.reshape(x, (batch_size, height // self.factor, self.factor, width // self.factor, self.factor, channels))
    out = tf.transpose(out, [0, 1, 3, 5, 2, 4])
    out = tf.reshape(out, (batch_size, height // self.factor, width // self.factor, channels * self.factor**2))

    return out

  def inverse(self, out):
    """

    :param out: (batch_size, height/factor, width/factor, factor^2*channels)
    :return x: (batch_size, height, width, channels)
    """

    (batch_size, height, width, channels) = self.out_shape
    x = tf.reshape(out, (batch_size, height, width, channels // self.factor**2, self.factor, self.factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (batch_size, height * self.factor, width * self.factor, channels // self.factor**2))

    return x

  def get_outshape(self):
    batch_size, height, width, channels = self.in_shape
    return batch_size, height // self.factor, width // self.factor, channels * self.factor**2
