from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def download_dataset(config):
  _ = tfds.load(config.dataset, data_dir=config.data_dir)


def train_transform(images, config):
  images = images["image"]
  if config.dataset == "mnist":
    paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
    images = tf.pad(images, paddings, "CONSTANT")
    images = tf.concat([images] * 3, axis=-1)
  elif config.dataset == "cifar10":
    paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
    images = tf.pad(images, paddings, "SYMMETRIC")
    images = tf.image.resize_with_crop_or_pad(images, 32, 32)
    images = tf.image.random_flip_left_right(images)

  images = tf.cast(images, tf.float32)
  images = images + tf.random.uniform(images.get_shape(), 0, 1)
  images = images / 256.
  images = images - 0.5

  return images


def test_transform(images, config):
  images = images["image"]
  if config.dataset == "mnist":
    paddings = tf.constant([[2, 2], [2, 2], [0, 0]])
    images = tf.pad(images, paddings, "CONSTANT")
    images = tf.concat([images] * 3, axis=-1)
  elif config.dataset == "cifar10":
    pass

  images = tf.cast(images, tf.float32)
  images = images + tf.random.uniform(images.get_shape(), 0, 1)
  images = images / 256.
  images = images - 0.5

  return images


def build_input_fns(config):

  def train_input_fn():
    dataset = tfds.load(config.dataset, split="train", data_dir=config.data_dir, shuffle_files=False)
    dataset = dataset.shuffle(50000).map(lambda x: train_transform(x, config)).repeat().batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tfds.load(config.dataset, split="test", data_dir=config.data_dir)
    dataset = dataset.map(lambda x: test_transform(x, config)).batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn


def build_fake_input_fns(config):
  random_sample = np.random.rand(*config.in_shape).astype("float32")

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).batch(config.batch_size).repeat()
    return tf.data.make_one_shot_iterator(dataset).get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        random_sample).batch(config.batch_size)
    return tf.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn
