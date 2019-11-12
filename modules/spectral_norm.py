from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def spectral_norm(w,
                  coeff,
                  power_iter,
                  debug=False):
  """

  :param w: conv weight of shape (k_height, k_width, c_in, c_out)
  :param coeff:
  :param power_iter:
  :param debug:
  :return:
  """

  w_shape = w.shape.as_list()

  # shape: (in_dim, c_out) where in_dim = k_height * k_width * c_in
  w_reshaped = tf.reshape(w, [-1, w_shape[-1]])

  # init u with shape (1, c_out)
  with tf.variable_scope("norm"):
    u_var = tf.get_variable("u",
                            shape=[1, w_shape[-1]],
                            initializer=tf.truncated_normal_initializer(),
                            trainable=False)
  u_normed = u_var
  v_normed = None

  for i in range(power_iter):
    # shape: (1, in_dim) = (1, c_out) x (c_out, in_dim)
    v = tf.matmul(u_normed, tf.transpose(w_reshaped))
    v_normed = tf.math.l2_normalize(v)

    # shape: (1, c_out) = (1, in_dim) x (in_dim, c_out)
    u = tf.matmul(v_normed, w_reshaped)
    u_normed = tf.math.l2_normalize(u)

  u_normed = tf.stop_gradient(u_normed)
  v_normed = tf.stop_gradient(v_normed)

  sigma = tf.matmul(tf.matmul(v_normed, w_reshaped), tf.transpose(u_normed))

  if debug:
    return sigma

  with tf.control_dependencies([u_var.assign(u_normed)]):
    factor = tf.maximum(tf.ones_like(w), sigma / coeff)
    w_normed = w / factor

  return w_normed


def spectral_norm_conv(w,
                       coeff,
                       power_iter,
                       in_shape,
                       out_shape,
                       stride,
                       padding,
                       debug=False):
  """

  :param w: conv weight of shape (k_height, k_width, c_in, c_out)
  :param coeff:
  :param power_iter:
  :param in_shape: (batch_size, height, width, in_channels)
  :param out_shape: (batch_size, height, width, out_channels)
  :param stride:
  :param padding:
  :param debug:
  :return:
  """

  # init u with shape: out_shape
  with tf.variable_scope("norm"):
    u_var = tf.get_variable("u_conv",
                            shape=out_shape,
                            initializer=tf.truncated_normal_initializer(),
                            trainable=False)

  u_normed = u_var
  v_normed = None

  for i in range(power_iter):
    # shape: (batch_size, height, width, in_channels)
    v = tf.nn.conv2d_transpose(u_normed,
                               filter=w,
                               output_shape=in_shape,
                               strides=stride,
                               padding=padding)
    v_normed = tf.math.l2_normalize(tf.reshape(v, [1, -1]))
    v_normed = tf.reshape(v_normed, v.shape)

    # shape: (batch_size, height, width, out_channels)
    u = tf.nn.conv2d(v_normed,
                     filter=w,
                     strides=stride,
                     padding=padding)
    u_normed = tf.math.l2_normalize(tf.reshape(u, [1, -1]))
    u_normed = tf.reshape(u_normed, u.shape)

  u_normed = tf.stop_gradient(u_normed)
  v_normed = tf.stop_gradient(v_normed)

  v_w = tf.nn.conv2d(v_normed,
                     filter=w,
                     strides=stride,
                     padding=padding)

  v_w = tf.reshape(v_w, [1, -1])

  sigma = tf.matmul(v_w, tf.reshape(u_normed, [-1, 1]))

  if debug:
    return sigma

  with tf.control_dependencies([u_var.assign(u_normed)]):
    factor = tf.maximum(tf.ones_like(w), sigma / coeff)
    w_normed = w / factor

  return w_normed
