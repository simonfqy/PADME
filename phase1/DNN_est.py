from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os
import sys
import pwd
import pdb
import grp

INPUT_PATH = "SimBoost/data/davis_cv_nmlzd.csv"
LOG_DIR = "logs/"
PATIENCE = 3

#uid = pwd.getpwnam("simonfqy").pw_uid
#gid = grp.getgrnam("def-cherkaso").gr_gid
#gid = grp.getgrnam("simonfqy").gr_gid

FEATURE_DIM = 98
FEATURES = ["d.n.obs","d.ave.sim","d.hist.sim1","d.hist.sim2","d.hist.sim3",
"d.hist.sim4","d.hist.sim5","d.hist.sim6","d.ave.val","d.num.nb","d.knn.sim1",
"d.knn.sim2","d.knn.sim3","d.knn.sim4","d.ave.knn1","d.ave.knn2","d.ave.knn3",
"d.ave.knn4","d.ave.knn5","d.ave.knn6","d.ave.knn7","d.ave.knn8","d.ave.knn9",
"d.w.ave.knn1","d.w.ave.knn2","d.w.ave.knn3","d.w.ave.knn4","d.w.ave.knn5",
"d.w.ave.knn6","d.w.ave.knn7","d.w.ave.knn8","d.w.ave.knn9","d.cl","d.bt",
"d.ev","d.pagerank","d.cl2","d.pr","d.mf1","d.mf2","d.mf3","d.mf4","d.mf5",
"d.mf6","d.mf7","d.mf8","d.mf9","d.mf10","t.n.obs","t.ave.sim","t.hist.sim1",
"t.hist.sim2","t.hist.sim3","t.hist.sim4","t.hist.sim5","t.hist.sim6","t.ave.val",
"t.num.nb","t.knn.sim1","t.knn.sim2","t.knn.sim3","t.knn.sim4","t.ave.knn1",
"t.ave.knn2","t.ave.knn3","t.ave.knn4","t.ave.knn5","t.ave.knn6","t.ave.knn7",
"t.ave.knn8","t.ave.knn9","t.w.ave.knn1","t.w.ave.knn2","t.w.ave.knn3","t.w.ave.knn4",
"t.w.ave.knn5","t.w.ave.knn6","t.w.ave.knn7","t.w.ave.knn8","t.w.ave.knn9","t.cl",
"t.bt","t.ev","t.pagerank","t.cl2","t.pr","t.mf1","t.mf2","t.mf3","t.mf4","t.mf5",
"t.mf6","t.mf7","t.mf8","t.mf9","t.mf10","d.t.ave","t.d.ave"]

train_obs = [24045, 24045, 24045, 24044, 24045]
test_obs = [6011, 6011, 6011, 6012, 6011]

def main():

  fold_index = [0]
  training_epochs = 39

  loaded_data = pd.read_csv(INPUT_PATH, dtype = np.float64, header=None)
  labels = (loaded_data.iloc[:,[98]]).values
  features = (loaded_data.iloc[:, range(98)]).values
  loaded_data = tf.contrib.data.Dataset.from_tensor_slices((features, labels))

  def _add_hidden_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)

  def my_model(features, labels, mode):
    
    feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    net = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      net = tf.layers.batch_normalization(net, center=False, scale=False, training=True)
      net = tf.layers.dropout(net, rate=0.5, training=True)
    else:
      net = tf.layers.batch_normalization(net, center=False, scale=False, training=False)
      net = tf.layers.dropout(net, rate=0.5, training=False)
    '''
    for units in [3]:      
      net = tf.layers.dense(net, units=units, activation=tf.nn.tanh)
      if mode == tf.estimator.ModeKeys.TRAIN:
        net = tf.layers.batch_normalization(net, center=False, scale=False, training=True)
      else:
        net = tf.layers.batch_normalization(net, center=False, scale=False, training=False)
    '''
    for layer_id, num_hidden_units in enumerate([30, 20, 10, 6]):
      with tf.variable_scope('hiddenlayer_%d' % layer_id, 
        values=(net,)) as hidden_layer_scope:
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
          name=hidden_layer_scope)
        _add_hidden_layer_summary(net, hidden_layer_scope.name)
        if mode == tf.estimator.ModeKeys.TRAIN:
          net = tf.layers.batch_normalization(net, center=False, scale=False, training=True)
          #net = tf.layers.dropout(net, rate=0.5, training=True)
        else:
          net = tf.layers.batch_normalization(net, center=False, scale=False, training=False)
          #net = tf.layers.dropout(net, rate=0.5, training=False)      
        
    '''
    if mode == tf.estimator.ModeKeys.TRAIN:
      net = tf.layers.dropout(net, rate=0.5, training=True)
    else:
      net = tf.layers.dropout(net, rate=0.5, training=False)
    '''
    outp = tf.layers.dense(net, 1, activation=None)
    predictions = tf.cast(tf.reshape(outp, [-1, 1]), tf.float64)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode, predictions={'value': predictions})

    loss = tf.losses.mean_squared_error(labels, predictions)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float64), predictions)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def dataset_input_fn(loaded_data, testing, perform_shuffle=False, num_epoch=1):
    
    def decode_line(features, label):      
      d = dict(zip(FEATURES, tf.split(features, num_or_size_splits=98))), label
      return d    

    base_skip = sum(train_obs[:(fold_index[0])]) + sum(test_obs[:
      (fold_index[0])])
    extra_skip = 0
    total_take = train_obs[(fold_index[0])]
    if testing:
      extra_skip = train_obs[(fold_index[0])]
      total_take = test_obs[(fold_index[0])]
    total_skip = base_skip + extra_skip

    print("total_take: {:d} \n".format(total_take))
    print("total_skip: {:d} \n".format(total_skip))

    dataset = (loaded_data.skip(total_skip).take(total_take).map(decode_line, 
      num_threads = 8, output_buffer_size=512)) 

    if perform_shuffle:
      dataset = dataset.shuffle(buffer_size=512)
      if num_epoch > 1:
        dataset_orig = dataset
        for _ in range(num_epoch - 1):
          dataset1 = dataset_orig.shuffle(buffer_size=512)
          dataset = dataset.concatenate(dataset1)          
    else:
      dataset = dataset.repeat(num_epoch)

    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

  def train_input_fn():
    return dataset_input_fn(loaded_data, False, perform_shuffle=True, num_epoch = 3)

  def test_input_fn():
    return dataset_input_fn(loaded_data, True)

  
  feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  
  regressor = tf.estimator.Estimator(model_fn = my_model, model_dir=LOG_DIR) 
  #log_file = open('log.txt', 'w+')
  
  for i in range(5):
    if tf.gfile.Exists(LOG_DIR):
      tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    log_file = open("log.txt", "a")   
    sys.stdout = log_file
    print("\nStarting fold number: {:d}".format(i+1))
    log_file.close()
    sys.stdout = sys.__stdout__
    best_rmse = 1000000000000
    wait_time = 0
    for j in range(int(round(training_epochs/3))):    
      regressor.train(input_fn=train_input_fn)
      ev = regressor.evaluate(input_fn=test_input_fn)
      #RMSE = np.sqrt(ev["average_loss"])
      RMSE = ev["rmse"]
      
      log_file = open("log.txt", "a")
      sys.stdout = log_file
      print("Epoch {:d}, RMSE: {:f}".format((j+1)*3 , RMSE))
      log_file.close()
      sys.stdout = sys.__stdout__
      if best_rmse >= RMSE:
        best_rmse = RMSE
        wait_time = 0
      else:
        wait_time += 1
        if (wait_time > PATIENCE):
          break

    fold_index[0] += 1
    
if __name__ == '__main__':
  main()
