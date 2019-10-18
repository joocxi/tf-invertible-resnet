from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from modules.squeeze import Squeeze


def test_squeeze():
  np.random.seed(2019)

  batch_size = 32
  input_size = 28
  channels = 3

  input_shape = (batch_size, input_size, input_size, channels)
  x_np = np.random.rand(*input_shape).astype("float32")
  x = tf.placeholder(tf.float32, [None, input_size, input_size, channels])

  squeeze = Squeeze(input_shape, 2)

  z = squeeze(x)
  x_inverse = squeeze.inverse(z)

  diff = tf.reduce_mean(tf.abs(x - x_inverse))

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _diff = sess.run(diff, feed_dict={x: x_np})
  print("Inversion difference is: {}".format(_diff))
