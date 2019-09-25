from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import Conv2D


class InvertibleBlock:
  def __init__(self,
               in_shape,
               stride,
               num_channel,
               coeff,
               power_iter,
               activation=tf.nn.relu,
               use_sn=True
               ):

    in_channel, height, width = in_shape
    self.layers = []
    self.layers.append(Conv2D(name="a",
                              in_channel=in_channel,
                              out_channel=num_channel,
                              kernel_size=3,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

    self.layers.append(activation)
    self.layers.append(Conv2D(name="b",
                              in_channel=num_channel,
                              out_channel=num_channel,
                              kernel_size=1,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

    self.layers.append(activation)
    self.layers.append(Conv2D(name="c",
                              in_channel=num_channel,
                              out_channel=in_channel,
                              kernel_size=3,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

  def __call__(self, x):
    shortcut = x
    for layer in self.layers:
      x = layer(x)
    return x + shortcut

  def inverse(self, out):
    x = out
    # TODO:
    return x
