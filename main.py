from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import train, train_test_split
from test import test_spectral_norm, test_trace_approximation

flags = tf.flags

# data config
flags.DEFINE_string("mode", "train", "Running mode: train/data/sn")
flags.DEFINE_string("dataset", "mnist", "The dataset to experiment with")

flags.DEFINE_string("save_dir", "model", "Model directory")
flags.DEFINE_string("data_dir", "data", "Data directory")

# training config
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("epochs", 10, "Num epochs")
flags.DEFINE_float("lr", 0.1, "Learning rate")

# invertible residual network config
flags.DEFINE_list("block_list", [7, 7, 7], "Block list")
flags.DEFINE_list("stride_list", [1, 2, 2], "Stride list")
flags.DEFINE_list("channel_list", [32, 64, 128], "Channel list")

flags.DEFINE_float("coeff", 0.9, "Scaling coefficient")
flags.DEFINE_integer("power_iter", 1, "Number of power iteration for spectral normalization")
flags.DEFINE_integer("num_trace_samples", 2, "Number of samples for Hutchinson trace estimator")
flags.DEFINE_integer("num_series_terms", 5, "Number of power series terms")


def main(_):
  config = flags.FLAGS
  if config.mode == "train":
    train(config)
  elif config.mode == "data":
    train_test_split(config)
  elif config.mode == "sn":
    test_spectral_norm()
  elif config.mode == "trace":
    test_trace_approximation()


if __name__ == "__main__":
  tf.compat.v1.app.run()
