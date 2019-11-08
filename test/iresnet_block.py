from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from scipy.linalg import logm

from modules import InvertibleBlock
from modules.spectral_norm import spectral_norm_conv
from test.spectral_norm import unfold_kernel


def test_trace_approximation():

  np.random.seed(2019)

  batch_size = 100
  in_dim = 3

  w_np = np.array([[1, 4, 6], [-0.5, -22, 7, ], [-4, 2, 3]])
  w_np = w_np / np.linalg.svd(w_np, compute_uv=False)[0] # enforcing Lipschitz constant < 1

  x_np = np.random.normal(size=(batch_size, in_dim))
  w = tf.constant(w_np, dtype=tf.float32)
  x = tf.constant(x_np, dtype=tf.float32)

  y = tf.matmul(x, w)

  # log(det(A)) = trace(ln(A))
  logdet = tf.linalg.logdet(w + tf.eye(in_dim))

  num_trace = 100
  num_series = 100

  u_shape = x.shape.as_list()
  u_shape.insert(1, num_trace)

  out = InvertibleBlock.compute_trace(
    x, y,
    num_power_series_terms=num_series,
    num_trace_samples=num_trace
  )

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _out, _logdet = sess.run([out, logdet])
  print("Mean of log determinant approximation: {}".format(np.mean(_out)))
  print("Log determinant by tf.linalg.logdet  : {}".format(_logdet))


def test_block_inversion():
  batch_size = 13
  height, width = 32, 32
  num_channel = 3
  block = InvertibleBlock(
    in_shape=(batch_size, height, width, num_channel),
    stride=2,
    num_channel=num_channel, # channel of intermediate layers
    coeff=0.97,
    power_iter=1,
    num_trace_samples=2,
    num_series_terms=2,
    activation=tf.nn.elu,
    use_sn=True,
    use_actnorm=False
  )

  x = tf.random.normal(shape=(batch_size, height, width, num_channel))

  out, trace = block(x)

  x_inverse = block.inverse(out)

  diff = tf.reduce_mean(tf.abs(x - x_inverse))

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _diff = sess.run(diff)
  print("Inversion difference is: {}".format(_diff))


def test_trace_sn():
  np.random.seed(2019)
  input_size = 7
  kernel_size = 3
  in_channel = 3
  out_channel = 3
  batch_size = 1

  w_np = np.random.normal(size=(kernel_size, kernel_size, in_channel, out_channel))
  x_np = np.random.normal(size=(batch_size, input_size, input_size, in_channel))

  w = tf.constant(w_np, dtype=tf.float32)
  x = tf.constant(x_np, dtype=tf.float32)

  in_shape = (batch_size, input_size, input_size, in_channel)
  out_shape = (batch_size, input_size, input_size, out_channel)

  w_sn = spectral_norm_conv(w, coeff=0.97, power_iter=15,
                            in_shape=in_shape, out_shape=out_shape,
                            stride=1, padding="SAME")

  Fx = tf.nn.conv2d(x, filter=w_sn, strides=1, padding="SAME")

  num_series = 100
  num_trace = 100

  tf.random.set_random_seed(2019)

  trace_approx = InvertibleBlock.compute_trace(
    x, Fx,
    num_power_series_terms=num_series,
    num_trace_samples=num_trace
  )

  grads = InvertibleBlock.compute_jacobian_matrix(x, Fx)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _trace_approx, _grads, _w_sn = sess.run([trace_approx, grads, w_sn])
  _grads = np.reshape(_grads, (_grads.shape[0], _grads.shape[1], -1))

  iden = np.eye(_grads.shape[1])

  # log_matrix(jac[i] + iden)
  log_jac = np.stack([logm(_grads[i] + iden) for i in range(batch_size)])

  # (batch_size)
  trace_jac = np.diagonal(log_jac, axis1=1, axis2=2).sum(1)

  # unfold kernel: y = conv2d(x, w) <=> y = x * w_unfolded
  w_unfolded = np.zeros([in_channel * (input_size ** 2),
                         out_channel * (input_size ** 2)])
  for c1 in range(in_channel):
    for c2 in range(out_channel):
      first_row = input_size * input_size * c1
      first_col = input_size * input_size * c2
      this_block = unfold_kernel(_w_sn[:, :, c1, c2], input_size)
      w_unfolded[first_row:(first_row + input_size * input_size),
      first_col:(first_col + input_size * input_size)] = this_block

  print("Trace approximation: {}".format(_trace_approx.sum()))
  print("Trace from full Jacobian : {}".format(trace_jac.sum()))
  print("Trace from unfolded matrix: {}".format(np.diagonal(logm(w_unfolded + iden)).sum()))
