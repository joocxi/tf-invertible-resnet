from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf
from tqdm import tqdm

from model import IResNet
from utils.data_utils import train_test_split


def train(config):
  train_data, test_data = train_test_split(config)

  train_data = train_data.repeat().shuffle(1024).batch(config.batch_size)
  train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle,
                                                 tf.data.get_output_types(train_data),
                                                 tf.data.get_output_shapes(train_data))

  train_iterator = tf.data.make_one_shot_iterator(train_data)
  val_iterator = tf.data.make_one_shot_iterator(test_data)

  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True

  x, _ = iterator.get_next()

  model = IResNet(in_shape=x.get_shape(),
                  block_list=config.block_list,
                  stride_list=config.stride_list,
                  channel_list=config.channel_list,
                  num_trace_samples=config.num_trace_samples,
                  num_series_terms=config.num_series_terms,
                  coeff=config.coeff,
                  power_iter=config.power_iter)

  z, loss = model(x)

  with tf.Session(config=sess_config) as sess:

    tf.set_random_seed(config.seed)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())

    train_op = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(loss)

    for global_step in tqdm(range(config.train_steps)):

      loss, train_op = sess.run([loss, train_op], feed_dict={handle: train_handle})

      if global_step % config.save_summary_period == 0:
        pass

        writer.flush()

      if global_step % config.save_model_period == 0:
        pass

    writer.close()


def evaluate(config):
  pass
