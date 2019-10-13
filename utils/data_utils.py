from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def download_dataset(config):
  _ = tfds.load(config.dataset, data_dir=config.data_dir)


def build_input_fns(config):

  def train_input_fn():
    dataset = tfds.load(config.dataset, split="train", data_dir=config.data_dir, shuffle_files=False)
    dataset = dataset.shuffle(50000).repeat().batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tfds.load(config.dataset, split="test", data_dir=config.data_dir)
    dataset = dataset.batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn


def build_fake_input_fns(config):
  random_sample = np.random.rand(config.batch_size, 28, 28, 1).astype("float32")

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).batch(config.batch_size).repeat()
    return tf.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn
