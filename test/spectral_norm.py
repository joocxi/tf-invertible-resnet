from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from modules.spectral_norm import spectral_norm, spectral_norm_conv


def unfold_kernel(kernel, in_shape):
  m = int((kernel.shape[0] - 1) / 2)

  transform_mat = np.zeros([in_shape ** 2, in_shape ** 2])
  for in_row in range(in_shape):
    for in_col in range(in_shape):
      mat_row = in_row * in_shape + in_col

      # Then, we need to parse values of the filter to each row
      for kernel_row in range(kernel.shape[0]):
        for kernel_col in range(kernel.shape[1]):
          r = in_row + kernel_row - m
          c = in_col + kernel_col - m

          if 0 <= r < in_shape and 0 <= c < in_shape:
            mat_col = r * in_shape + c
            transform_mat[mat_row, mat_col] = kernel[kernel_row, kernel_col]

  return transform_mat.T


def test_spectral_norm():
  np.random.seed(2019)

  input_size = 7
  kernel_size = 3
  in_channel = 2
  out_channel = 6

  # for easy testing, we assume outputs of convolution have the same spatial
  # dimensions as inputs, so do not change the `stride` and `padding` here
  stride = 1
  padding = "SAME"

  coeff = 0.9
  power_iter = 100

  w_np = np.random.normal(size=(kernel_size, kernel_size, in_channel, out_channel))
  x_np = np.random.normal(size=(1, input_size, input_size, in_channel))
  w = tf.constant(w_np, dtype=tf.float32)
  x = tf.constant(x_np, dtype=tf.float32)

  # spectral norm of reshaped kernel
  with tf.variable_scope("sigma"):
    sigma = spectral_norm(w, coeff, power_iter, debug=True)

  # spectral norm of unfolded kernel
  with tf.variable_scope("sigma_conv"):
    sigma_conv = spectral_norm_conv(w,
                                    coeff,
                                    power_iter,
                                    in_shape=x.shape,
                                    out_shape=(1, input_size, input_size, out_channel),
                                    stride=1,
                                    padding=padding,
                                    debug=True)

  # create y to check unfolded kernel
  y = tf.nn.conv2d(
    x,
    filter=w,
    strides=stride,
    padding=padding,
  )

  # svd of reshaped kernel
  w_reshaped = tf.reshape(w, [-1, w.shape[-1]])
  s_reshaped = tf.svd(w_reshaped, compute_uv=False)[0]

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  _sigma, _sigma_conv, _s_reshaped, _y = sess.run([sigma, sigma_conv, s_reshaped, y])

  # unfold kernel: y = conv2d(x, w) <=> y = x * w_unfolded
  w_unfolded = np.zeros([in_channel * (input_size ** 2),
                         out_channel * (input_size ** 2)])
  for c1 in range(in_channel):
    for c2 in range(out_channel):
      first_row = input_size * input_size * c1
      first_col = input_size * input_size * c2
      this_block = unfold_kernel(w_np[:, :, c1, c2], input_size)
      w_unfolded[first_row:(first_row + input_size * input_size),
      first_col:(first_col + input_size * input_size)] = this_block

  x_np_reshaped = np.reshape(np.transpose(x_np, axes=[0, 3, 1, 2]), [1, -1])
  y_unfolded = np.dot(x_np_reshaped, w_unfolded)
  _y_reshaped = np.reshape(np.transpose(_y, axes=[0, 3, 1, 2]), [1, -1])

  print("Mean Absolute Error between conv2d(x, w) and x * w_unfolded : {}.".\
        format(np.mean(np.abs(_y_reshaped - y_unfolded))))

  print("Largest singular value of reshaped kernel by tf.svd: {}.".format(_s_reshaped))
  print("Largest singular value of reshaped kernel estimated after {} iteration: {}.".\
        format(power_iter, _sigma[0, 0]))

  s_unfolded = np.linalg.svd(w_unfolded, compute_uv=False)[0]
  print("Largest singular value of unfolded kernel by np.linalg.svd: {}.".format(s_unfolded))
  print("Largest singular value of unfolded kernel estimated after {} iteration: {}.".\
        format(power_iter, _sigma_conv[0, 0]))
