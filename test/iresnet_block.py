from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


from modules import InvertibleBlock


def test_trace_approximation():

  np.random.seed(2019)

  batch_size = 100
  in_dim = 3

  w_np = np.array([[1, 0, 0], [0, 2, 0,], [0, 0, 3]])
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

  out = InvertibleBlock.compute_trace(x, y,
                                      num_power_series_terms=num_series,
                                      num_trace_samples=num_trace)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _out, _logdet = sess.run([out, logdet])
  print("Mean of log determinant approximation: {}".format(np.mean(_out)))
  print("Log determinant by tf.linalg.logdet  : {}".format(_logdet))
