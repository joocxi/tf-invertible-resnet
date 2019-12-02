from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v1 as tf

from PIL import Image

from model import IResNet
from utils.data_utils import build_input_fns, build_fake_input_fns


def model_fn(features, labels, mode, params, config):
  del labels, config

  model = IResNet(in_shape=params.in_shape,
                  block_list=params.block_list,
                  stride_list=params.stride_list,
                  channel_list=params.channel_list,
                  num_trace_samples=params.num_trace_samples,
                  num_series_terms=params.num_series_terms,
                  coeff=params.coeff,
                  power_iter=params.power_iter)

  if params.mode == "generate":
    predictions = model.sample(params.batch_size)
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions
    )

  z, log_prob_z, trace, loss = model(features)

  if params.mode == "reconstruct":
    predictions = model.inverse(z)
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions
    )

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(
    params.learning_rate, global_step, params.train_steps)

  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  logging_hook = tf.train.LoggingTensorHook(
    {
      "loss": loss,
      "trace": trace,
      "log_prob_z" : log_prob_z
    },
    every_n_iter=10
  )

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    training_hooks=[logging_hook],
  )


def train(config, debug=False):

  if config.delete_existing and tf.io.gfile.exists(config.checkpoint_dir):
    tf.logging.warn("Deleting old log directory at {}".format(
        config.checkpoint_dir))
    tf.io.gfile.rmtree(config.checkpoint_dir)
  tf.io.gfile.makedirs(config.checkpoint_dir)

  if debug:
    train_input_fn, eval_input_fn = build_fake_input_fns(config)
  else:
    train_input_fn, eval_input_fn = build_input_fns(config)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=config,
    config=tf.estimator.RunConfig(
      model_dir=config.checkpoint_dir,
      save_checkpoints_steps=config.viz_steps,
    ),
  )

  for _ in range(config.train_steps // config.viz_steps):
    estimator.train(train_input_fn, steps=config.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)


def generate(config):
  gen_dir = "generated"
  if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=config,
    model_dir=config.checkpoint_dir
  )

  _, input_fn = build_fake_input_fns(config)
  for pred in estimator.predict(input_fn, yield_single_examples=False):
    for i in range(config.batch_size):
      arr = pred[i]
      arr = np.clip(arr, -0.5, 0.5)
      arr = arr + 0.5
      arr = (arr * 255).astype("uint8")
      im = Image.fromarray(arr)
      im.save(os.path.join(gen_dir, "image_{}.png".format(i)))
    break


def reconstruct(config):
  res_folder = "reconstructed"
  if not os.path.exists(res_folder):
    os.mkdir(res_folder)

  estimator = tf.estimator.Estimator(
    model_fn,
    params = config,
    model_dir = config.checkpoint_dir
  )

  _, input_fn = build_input_fns(config)

  for pred in estimator.predict(input_fn, yield_single_examples=False):
    for i in range(config.batch_size):
      arr = pred[i]
      arr = np.clip(arr, -0.5, 0.5)
      arr = arr + 0.5
      arr = (arr * 255).astype("uint8")
      im = Image.fromarray(arr)
      im.save(os.path.join(res_folder, "image_{}.png".format(i)))
    break
