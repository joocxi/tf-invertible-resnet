from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np

from modules import InvertibleBlock, Squeeze


class IResNet:
  def __init__(self,
               in_shape,
               block_list,
               stride_list,
               channel_list,
               num_trace_samples,
               num_series_terms,
               coeff,
               power_iter,
               init_squeeze=True):
    """

    :param in_shape: (batch_size, height, width, in_channel)
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
    self.in_shape = in_shape

    if init_squeeze:
      self.squeeze = Squeeze(in_shape, 2)
      in_shape = self.squeeze.out_shape # update in_shape after squeezing
    else:
      self.squeeze = None

    self.blocks = []
    for idx, (num_block, stride, num_channel) in \
        enumerate(zip(block_list, stride_list, channel_list)):
      with tf.variable_scope('stack%d' % (idx + 1)):
        in_shape = self.create_stack(num_block, stride, num_channel, in_shape)

    in_dim = np.prod(self.in_shape[1:])
    self.batch_size = self.in_shape[0]

    with tf.variable_scope('prior'):
      loc = tf.get_variable("loc", shape=[in_dim], initializer=tf.constant_initializer(0))
      log_scale = tf.get_variable("scale", shape=[in_dim], initializer=tf.constant_initializer(0))
      self.prior = tfp.distributions.Normal(loc=loc, scale=tf.exp(log_scale))

  def __call__(self, x):

    if self.squeeze is not None:
      x = self.squeeze(x)

    z = x
    traces = []
    for block in self.blocks:
      z, trace = block(z)
      traces.append(trace)

    # (batch_size,)
    trace = tf.add_n(traces)
    log_prob_z = self.log_prob(z)

    log_prob_x = log_prob_z + trace

    loss = - log_prob_x / float(np.log(2.) * np.prod(self.in_shape[1:])) + 8

    return z, tf.reduce_mean(loss)

  def inverse(self, out):
    x = out

    for block in reversed(self.blocks):
      x = block.inverse(x)

    if self.squeeze is not None:
      return self.squeeze.inverse(x)

    return x

  def create_stack(self,
                   num_block,
                   stride,
                   num_channel,
                   in_shape):

    for idx in range(num_block):
      with tf.variable_scope('block%d' % (idx + 1)):
        block_stride = stride if idx == 0 else 1 # apply stride only to the first block of each stacks
        self.blocks.append(InvertibleBlock(in_shape,
                                           block_stride,
                                           num_channel,
                                           self.coeff,
                                           self.power_iter,
                                           self.num_trace_samples,
                                           self.num_series_terms))
        if block_stride == 2:
          in_shape = Squeeze.get_outshape(in_shape, stride)

    return in_shape

  def log_prob(self, z):

    z_reshaped = tf.reshape(z, (self.batch_size, -1))
    log_prob = self.prior.log_prob(z_reshaped)
    log_prob = tf.reduce_sum(log_prob, axis=-1)
    return log_prob
