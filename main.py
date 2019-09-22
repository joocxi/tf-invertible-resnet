from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import train, train_test_split

flags = tf.flags

flags.DEFINE_string("mode", "train", "Running mode: train/download_data")
flags.DEFINE_string("dataset", "mnist", "The dataset to experiment with")

flags.DEFINE_string("save_dir", "model", "Model directory")
flags.DEFINE_string("data_dir", "data", "Data directory")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("epochs", 10, "Num epochs")
flags.DEFINE_float("lr", 0.1, "Learning rate")

def main(_):
  config = flags.FLAGS
  if config.mode == "train":
    train(config)
  elif config.mode == "debug":
    train_test_split(config)


if __name__ == "__main__":
  tf.compat.v1.app.run()
