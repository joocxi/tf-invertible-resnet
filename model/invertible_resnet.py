from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import InvertibleBlock


class IResNet:
  def __init__(self,
               in_shape,
               block_list,
               stride_list,
               channel_list,
               num_trace_samples,
               num_series_terms,
               coeff,
               power_iter):

    assert len(block_list) == len(stride_list) == len(channel_list)
    self.coeff = coeff
    self.power_iter = power_iter

    self.blocks = []
    for idx, (num_block, stride, num_channel) in \
        enumerate(zip((block_list, stride_list, channel_list))):
      with tf.variable_scope('stack%d' % (idx + 1)):
        self.create_stack(num_block, stride, num_channel, in_shape)

  def __call__(self, x):

    for block in self.blocks:
      x = block(x)

    return x

  def create_stack(self,
                   num_block,
                   stride,
                   num_channel,
                   in_shape):

    for idx in range(num_block):
      with tf.variable_scope('block%d' % (idx + 1)):
        self.blocks.append(InvertibleBlock(in_shape,
                                           stride,
                                           num_channel,
                                           self.coeff,
                                           self.power_iter))

  def inverse(self, out):
    x = out
    # TODO:
    return x
