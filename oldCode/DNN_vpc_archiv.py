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

INPUT_PATH = "SimBoost/data/davis_cv2.csv"
LOG_DIR = "logs/"

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

train_obs = [16000, 3200, 3200, 24045, 24045, 24044, 24045]
test_obs = [4800, 6011, 6011, 6012, 6011]
fold_index = [0]

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
    #print("Performed shuffling. \n")
  #print("Not performed shuffling. \n")
  else:
    dataset = dataset.repeat(num_epoch)

  dataset = dataset.batch(32)
  return dataset

def main():
  
  if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
  tf.gfile.MakeDirs(LOG_DIR)
  #os.chown(LOG_DIR, uid, gid)  

  loaded_data = pd.read_csv(INPUT_PATH, dtype = np.float64, header=None)
  labels = (loaded_data.iloc[:,[98]]).values
  features = (loaded_data.iloc[:, range(98)]).values
  loaded_data = tf.contrib.data.Dataset.from_tensor_slices((features, labels))

  def my_model(features, labels, mode):    
    feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    net = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.05)
    for units in [3]:
      if mode == tf.estimator.ModeKeys.TRAIN:
        net = tf.layers.dropout(net, rate=0.5, training=True)
      else:
        net = tf.layers.dropout(net, rate=0.5, training=False)      
      net = tf.layers.dense(net, units=units, activation=tf.nn.tanh
                            #,kernel_regularizer=regularizer
                            )
    
    outp = tf.layers.dense(net, 1, activation=None
                           #, kernel_regularizer=regularizer
                           )
    predictions = tf.cast(tf.reshape(outp, [-1, 1]), tf.float64)

    #if mode == tf.estimator.ModeKeys.PREDICT:
    #  return tf.estimator.EstimatorSpec(mode, predictions={'value': predictions})

    loss = tf.losses.mean_squared_error(labels, predictions)
    eval_metric_ops = {
      #"rmse": tf.metrics.root_mean_squared_error(
      #  tf.cast(labels, tf.float64), predictions)
       "rmse": tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
    }
    return loss, eval_metric_ops     

  def train_input_fn():
    #fold_index[0] += 1
    dts = dataset_input_fn(loaded_data, False, perform_shuffle=False, num_epoch = 1)
    return dts

  def test_input_fn():
    dts = dataset_input_fn(loaded_data, True)
    return dts
  
  #feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  def get_train_op(tr_bch_feat, tr_bch_labl):    
    #train_itr = train_data.make_one_shot_iterator()  
    #tr_bch_feat, tr_bch_labl = train_itr.get_next()
    loss, _ = my_model(tr_bch_feat, tr_bch_labl, tf.estimator.ModeKeys.TRAIN)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return train_op

  train_data = train_input_fn()
  validn_data = test_input_fn()

  
  handle = tf.placeholder(tf.string, shape=[])
  #mode = tf.placeholder(tf.string, shape=[])
  
  iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, train_data.output_types, train_data.output_shapes)
  features, labels = iterator.get_next()
  

  #def my_model(features, labels, mode): 
  '''   
  feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  net = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
  
  #regularizer = tf.contrib.layers.l2_regularizer(scale=0.05)

  for units in [3]:
    if mode == "train":
      net = tf.layers.dropout(net, rate=0.5, training=True)
    else:
      net = tf.layers.dropout(net, rate=0.5, training=False)      
    net = tf.layers.dense(net, units=units, activation=tf.nn.tanh
                          #,kernel_regularizer=regularizer
                          )
  
  outp = tf.layers.dense(net, 1, activation=None
                         #, kernel_regularizer=regularizer
                         )
  predictions = tf.cast(tf.reshape(outp, [-1, 1]), tf.float64)

  #if mode == tf.estimator.ModeKeys.PREDICT:
  #  return tf.estimator.EstimatorSpec(mode, predictions={'value': predictions})

  loss = tf.losses.mean_squared_error(labels, predictions)
  eval_metric_ops = {
    #"rmse": tf.metrics.root_mean_squared_error(
    #  tf.cast(labels, tf.float64), predictions)
    "rmse": tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
  }
  #return loss, eval_metric_ops
  '''
  train_itr = train_data.make_one_shot_iterator()
  validn_itr = validn_data.make_initializable_iterator()

  #graph_train = tf.Graph()
  #with graph_train.as_default():
  
  #train_itr = tf.contrib.data.Iterator.from_structure(train_data.output_types, 
  #  train_data.output_shapes)
  '''
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  '''
  train_op = get_train_op(features, labels)
  #train_init_op = train_itr.make_initializer(train_data)

  #TODO: if this doesn't work, try feedable iterator.

  #graph_eval = tf.Graph()
  #with graph_eval.as_default():
  
  #validn_itr = tf.contrib.data.Iterator.from_structure(train_data.output_types, 
  #  train_data.output_shapes)
  #vl_bch_feat, vl_bch_labl = validn_itr.get_next()
  _, eval_metric_ops = my_model(features, labels, tf.estimator.ModeKeys.EVAL)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    training_handle = sess.run(train_itr.string_handle())
    validation_handle = sess.run(validn_itr.string_handle())
    #sess.run(train_init_op)
    #log_file = open('log.txt', 'w+')    
    for i in range(1):
      #if i > 0:
        #if tf.gfile.Exists(LOG_DIR):
        #  tf.gfile.DeleteRecursively(LOG_DIR)
        #tf.gfile.MakeDirs(LOG_DIR)
        #log_file = open("log.txt", "a")
      #sys.stdout = log_file
      sess.run(validn_itr.initializer)
      for step in range(500):
        try:
          sess.run(train_op, feed_dict={handle: training_handle})
            #, mode: "train"})
        except tf.errors.OutOfRangeError:
          print("Reached the end of the data set. \n")
          '''
          fold_index[0] += 1
          train_data = train_input_fn()
          train_itr = tf.contrib.data.Iterator.from_structure(train_data.output_types, 
            train_data.output_shapes)
          train_op = get_train_op(train_data, train_itr)
          train_init_op = train_itr.make_initializer(train_data)
          sess.run(train_init_op)
          sess.run(train_op)
          '''
        if step % 500 == 499:
          summ = 0
          count = 0
          for i in range(150):
            ev = sess.run(eval_metric_ops, feed_dict={handle: validation_handle})
              #, mode: "eval"})
            #RMSE = np.sqrt(ev["average_loss"])
            RMSE = ev["rmse"]
            summ += RMSE
            count = i + 1
          mean = summ/count
          print("RMSE: {0:f}".format(mean))
      #log_file.close()
      #sys.stdout = sys.__stdout__

if __name__ == '__main__':
  main()

"""
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
"""
