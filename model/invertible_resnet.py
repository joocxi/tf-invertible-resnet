from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

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
    """

    :param in_shape: (height, width, in_channel)
    :param block_list:
    :param stride_list:
    :param channel_list:
    :param num_trace_samples:
    :param num_series_terms:
    :param coeff:
    :param power_iter:
    """

    assert len(block_list) == len(stride_list) == len(channel_list)
    self.coeff = coeff
    self.power_iter = power_iter
    self.num_trace_samples = num_trace_samples
    self.num_series_terms = num_series_terms

    self.blocks = []
    for idx, (num_block, stride, num_channel) in \
        enumerate(zip((block_list, stride_list, channel_list))):
      with tf.variable_scope('stack%d' % (idx + 1)):
        self.create_stack(num_block, stride, num_channel, in_shape)

    # TODO:
    in_dim = None
    self.prior = tfp.distributions.Normal(loc=tf.zeros(in_dim), scale=tf.zeros(in_dim))

  def __call__(self, x):

    z = x
    traces = []
    for block in self.blocks:
      z, trace = block(z)
      traces.append(trace)

    # (batch_size,)
    trace = tf.add_n(traces)
    log_prob_z = self.log_prob(z)

    log_prob_x = log_prob_z + trace

    return z, log_prob_x

  def inverse(self, out):
    x = out

    for block in reversed(self.blocks):
      x = block.inverse(x)
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
                                           self.power_iter,
                                           self.num_trace_samples,
                                           self.num_series_terms))

  def log_prob(self, z):
    return self.prior.log_prob(z)
