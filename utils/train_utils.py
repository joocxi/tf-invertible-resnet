from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf

from model import IResNet
from utils.data_utils import build_input_fns, build_fake_input_fns


def model_fn(features, labels, mode, params, config):
  del labels, config

  model = IResNet(in_shape= (32, 28, 28, 1),
                  block_list=params.block_list,
                  stride_list=params.stride_list,
                  channel_list=params.channel_list,
                  num_trace_samples=params.num_trace_samples,
                  num_series_terms=params.num_series_terms,
                  coeff=params.coeff,
                  power_iter=params.power_iter)

  z, loss = model(features)

  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(
    params.learning_rate, global_step, params.train_steps)

  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op
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


def evaluate(config):
  pass
