from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds


def train_test_split(config):
  train_split = tfds.load(config.dataset, split="train", data_dir=config.data_dir)
  test_split = tfds.load(config.dataset, split="test", data_dir=config.data_dir)
  return train_split, test_split
