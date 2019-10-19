from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from modules.conv2d import Conv2D
from modules.squeeze import Squeeze


class InvertibleBlock:
  def __init__(self,
               in_shape,
               stride,
               num_channel,
               coeff,
               power_iter,
               num_trace_samples,
               num_series_terms,
               activation=tf.nn.elu,
               use_sn=True,
               use_actnorm=True
               ):
    """

    :param in_shape: (batch_size, height, width, channels)
    :param stride:
    :param num_channel:
    :param coeff:
    :param power_iter:
    :param num_trace_samples:
    :param num_series_terms:
    :param activation:
    :param use_sn:
    :param use_actnorm:
    """

    self.num_trace_samples = num_trace_samples
    self.num_series_terms = num_series_terms

    if stride == 2:
      self.squeeze = Squeeze(in_shape, stride)
      in_shape = self.squeeze.out_shape # update in_shape after squeezing
    else:
      self.squeeze = None

    batch_size, height, width, channels = in_shape

    self.layers = []
    self.layers.append(Conv2D(name="a",
                              in_shape=in_shape,
                              in_channel=channels,
                              out_channel=num_channel, # intermediate channels
                              kernel_size=3,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

    self.layers.append(activation)
    int_shape = (batch_size, height, width, num_channel)
    self.layers.append(Conv2D(name="b",
                              in_shape=int_shape,
                              in_channel=num_channel,
                              out_channel=num_channel,
                              kernel_size=1,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

    self.layers.append(activation)
    self.layers.append(Conv2D(name="c",
                              in_shape=int_shape,
                              in_channel=num_channel,
                              out_channel=channels,
                              kernel_size=3,
                              use_sn=use_sn,
                              coeff=coeff,
                              power_iter=power_iter))

    if use_actnorm:
      # TODO:
      pass


  def residual_block(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def __call__(self, x, ignore_logdet=False):
    """

    :param x: (batch_size, height, width, num_channel)
    :param ignore_logdet: boolean
    :return trace: (batch_size,)
    """

    if self.squeeze is not None:
      x = self.squeeze(x)

    Gx = self.residual_block(x)

    trace = self.compute_trace(x, Gx,
                               self.num_trace_samples,
                               self.num_series_terms)

    if ignore_logdet:
      return x + Gx, None
    else:
      return x + Gx, trace

  def inverse(self, out, fixed_point_iter=100):
    x = out
    for i in range(fixed_point_iter):
      x = out - self.residual_block(x)

    if self.squeeze is not None:
      return self.squeeze.inverse(x)

    return x

  @staticmethod
  def compute_trace(x, Gx,
                    num_trace_samples=2,
                    num_power_series_terms=2):

    u_shape = tf.shape(x)
    u_shape = tf.concat([u_shape, [num_trace_samples]], axis=0)

    # shape (batch_size, height, width, num_channel, num_sample)
    u = tf.random.normal(u_shape)

    def loop_trace_samples(n, trace_total):

      u_reshaped = tf.reshape(u[..., n], (u_shape[0], -1, 1))

      def loop_series_terms(k, output_grads, trace):
        """

        :param k:
        :param output_grads: (batch_size, height, width, num_channel)
        :param trace: (batch_size,)
        :return:
        """
        grads = tf.gradients(Gx, x, output_grads)[0]

        grads_reshaped = tf.reshape(grads, (u_shape[0], 1, -1))


        trace = trace + tf.squeeze(tf.cond(tf.equal(k % 2, 0), lambda: 1.0, lambda: -1.0) *\
                             tf.matmul(grads_reshaped, u_reshaped) / tf.cast(k + 1, tf.float32), axis= [1, 2])
        return k + 1, grads, trace

      _, _, trace_by_sample = tf.while_loop(
        cond=lambda k, _1, _2: k < num_power_series_terms,
        body=loop_series_terms,
        loop_vars=[tf.constant(0, dtype=tf.int32), u[..., n], tf.zeros(shape=u_shape[0])]
      )

      return n + 1, trace_total + trace_by_sample

    _, trace_all_samples = tf.while_loop(
      cond=lambda n, _: n < num_trace_samples,
      body=loop_trace_samples,
      loop_vars = [tf.constant(0, dtype=tf.int32), tf.zeros(shape=u_shape[0])]
    )

    trace_all_samples = trace_all_samples / num_trace_samples
    return trace_all_samples
