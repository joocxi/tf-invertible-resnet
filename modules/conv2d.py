from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from modules.spectral_norm import spectral_norm, spectral_norm_conv


class Conv2D:
  def __init__(self,
               name,
               in_channel,
               out_channel,
               kernel_size,
               stride=1,
               padding="SAME",
               use_bias=True,
               use_sn=False,
               coeff=0.97,
               power_iter=5
               ):

    self.stride = stride
    self.padding=padding
    self.use_bias = use_bias
    self.use_sn = use_sn

    weight_shape = [kernel_size, kernel_size, in_channel, out_channel]
    weight_initializer = tf.initializers.truncated_normal()

    with tf.variable_scope('conv%s' % name):
      self.weight = tf.get_variable("weight", shape=weight_shape, initializer=weight_initializer)
      self.bias = tf.get_variable("bias", [out_channel], initializer=tf.constant_initializer(0.0))

    if self.use_sn:
      if kernel_size == 1:
        self.weight_sn = spectral_norm(self.weight, coeff, power_iter)
      else:
        self.weight_sn = spectral_norm_conv(self.weight, coeff, power_iter,
                                            in_shape=None, out_shape=None,
                                            stride=stride, padding=padding)

  def __call__(self, x):
    if self.use_sn:
      x =  tf.nn.conv2d(x, filter=self.weight_sn, strides=self.stride, padding=self.padding)
    else:
      x =  tf.nn.conv2d(x, filter=self.weight, strides=self.stride, padding=self.padding)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.bias)

    return x
