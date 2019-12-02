from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from utils import train, download_dataset, generate, reconstruct
from test import test_spectral_norm, test_trace_approximation,\
  test_block_inversion, test_iresnet, test_squeeze, test_trace_sn

flags = tf.flags

# data config
flags.DEFINE_string("mode", "train", "Running mode: train/data/sn")
flags.DEFINE_string("dataset", "mnist", "The dataset to experiment with")

flags.DEFINE_string("checkpoint_dir", "ckpt", "Model directory")
flags.DEFINE_string("data_dir", "data", "Data directory")

# training config
flags.DEFINE_integer("seed", 2019, "Random seed")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("epochs", 10, "Num epochs")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
flags.DEFINE_float('weight_decay', 5e-4, "Coefficient for weight decay")
flags.DEFINE_integer("train_steps", 100000, "Num training steps")
flags.DEFINE_integer("viz_steps", 5000, "Num steps at which do visualization")
flags.DEFINE_integer("log_steps", 50, "Num steps to print out loss")


# invertible residual network config
flags.DEFINE_list("in_shape", [], "Input shape")
flags.DEFINE_list("block_list", [7, 7, 7], "Block list")
flags.DEFINE_list("stride_list", [1, 2, 2], "Stride list")
flags.DEFINE_list("channel_list", [32, 64, 128], "Channel list")

flags.DEFINE_float("coeff", 0.9, "Scaling coefficient")
flags.DEFINE_integer("power_iter", 1, "Number of power iteration for spectral normalization")
flags.DEFINE_integer("num_trace_samples", 2, "Number of samples for Hutchinson trace estimator")
flags.DEFINE_integer("num_series_terms", 5, "Number of power series terms")

# TensorBoard config
flags.DEFINE_integer("save_summary_period", 100, "")
flags.DEFINE_integer("save_model_period", 100, "")
flags.DEFINE_bool("delete_existing", True, "")


def main(_):
  config = flags.FLAGS
  if config.mode == "train":
    assert config.dataset in ("mnist", "cifar10")
    config.in_shape = (config.batch_size, 32, 32, 3)
    config.block_list = [eval(x) for x in config.block_list]
    config.stride_list = [eval(x) for x in config.stride_list]
    config.channel_list = [eval(x) for x in config.channel_list]

    train(config)
  elif config.mode == "debug":
    config.train_steps = 1
    config.viz_steps = 1
    config.block_list = [2, 2, 2]
    config.channel_list = [3, 4, 5]
    config.stride_list = [1, 1, 2]
    config.in_shape = (config.batch_size, 28, 28, 1)
    train(config, debug=True)
  elif config.mode == "prepare":
    download_dataset(config)
  elif config.mode == "sn":
    test_spectral_norm()
  elif config.mode == "iresnet":
    test_iresnet()
  elif config.mode == "trace":
    test_trace_approximation()
  elif config.mode == "inverse":
    test_block_inversion()
  elif config.mode == "squeeze":
    test_squeeze()
  elif config.mode == "trace_sn":
    test_trace_sn()
  elif config.mode == "generate":
    generate(config)
  elif config.mode == "reconstruct":
    reconstruct(config)


if __name__ == "__main__":
  tf.app.run()
