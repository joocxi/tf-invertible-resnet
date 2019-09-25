from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def spectral_norm(weight, coeff, power_iter):
  u = tf.get_variable("weight_u",
                      [1, weight[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
