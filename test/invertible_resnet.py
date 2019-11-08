from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from model import IResNet


def test_iresnet():

  np.random.seed(2019)

  batch_size = 2
  input_size = 8
  channels = 3

  input_shape = (batch_size, input_size, input_size, channels)
  x_np = np.random.rand(*input_shape).astype("float32")
  x = tf.placeholder(tf.float32, [None, input_size, input_size, channels])

  net = IResNet(
    in_shape=input_shape,
    block_list=[1, 2, 3],
    stride_list=[1, 2, 2],
    channel_list=[2, 3, 5],
    num_trace_samples=4,
    num_series_terms=3,
    coeff=0.97,
    power_iter=2)

  log_prob_z, trace, loss = net(x)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _log_prob_z, _trace, _loss = sess.run([log_prob_z, trace, loss], feed_dict={x: x_np})
  print("Loss is: {}".format(_loss))
  print("Trace is: {}".format(_trace))
  print("Log prob is: {}".format(_log_prob_z))
