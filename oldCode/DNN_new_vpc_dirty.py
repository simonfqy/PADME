from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import math
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
# signals whether we are doing early stopping or not.
VALIDATING = True
#VALIDATING = False
# The ratio of the validation set in the training set.
VALDN_RTO = 0.1
BATCH_SIZE = 32
PATIENCE = 3

train_obs = [3200, 3200, 3200, 24045, 24045, 24045, 24044, 24045]
test_obs = [640, 640, 640, 6011, 6011, 6011, 6012, 6011]
fold_index = [0]

def dataset_input_fn(loaded_data, testing, perform_shuffle=False, num_epoch=1, 
  validating=VALIDATING, fold_index=fold_index):
  ''' If testing + validating, generates the validation data set.
      If testing + not validating, generates the test set for that epoch.
      If not testing + validating, generates the (smaller) training data set
      If not testing + not validating, generates the (bigger) training data set. 
  '''
  def decode_line(features, label):      
    d = dict(zip(FEATURES, tf.split(features, num_or_size_splits=98))), label
    return d
  
  base_skip = sum(train_obs[:fold_index[0]]) + sum(test_obs[:fold_index[0]])
  extra_skip = 0

  if not validating:    
    total_take = train_obs[fold_index[0]]
    if testing:
      extra_skip = train_obs[fold_index[0]]
      total_take = test_obs[fold_index[0]]
  else:
    total_take = math.floor((1 - VALDN_RTO) * train_obs[fold_index[0]])
    if testing:
      extra_skip = math.floor((1 - VALDN_RTO) * train_obs[fold_index[0]])
      total_take = math.ceil(VALDN_RTO * train_obs[fold_index[0]])

  total_skip = base_skip + extra_skip

  print("fold_index: {:d} \n".format(fold_index[0]))
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

  dataset = dataset.batch(BATCH_SIZE)
  return dataset, total_take * num_epoch

def main():
  
  if tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
  tf.gfile.MakeDirs(LOG_DIR)
  #os.chown(LOG_DIR, uid, gid)  

  loaded_data = pd.read_csv(INPUT_PATH, dtype = np.float64, header=None)
  labels = (loaded_data.iloc[:,[98]]).values
  features = (loaded_data.iloc[:, range(98)]).values
  loaded_data = tf.contrib.data.Dataset.from_tensor_slices((features, labels))
  '''
  def train_input_fn():
    #fold_index[0] += 1
    dts = dataset_input_fn(loaded_data, False, perform_shuffle=False, num_epoch = 1)
    return dts

  def test_input_fn():
    dts = dataset_input_fn(loaded_data, True)
    return dts
  '''
  with tf.Session() as sess:
    
    #log_file = open('log.txt', 'w+')
    # For each fold in the cross validation.    
    for i in range(3):
      #if i > 0:
        #if tf.gfile.Exists(LOG_DIR):
        #  tf.gfile.DeleteRecursively(LOG_DIR)
        #tf.gfile.MakeDirs(LOG_DIR)
        #log_file = open("log.txt", "a")
      #sys.stdout = log_file
      handle = tf.placeholder(tf.string, shape=[])
      mode = tf.placeholder(tf.string, shape=[])
      train_data, train_size = dataset_input_fn(loaded_data, False, 
        perform_shuffle=False, num_epoch = 2)
      # It could be either validation data set or test data set, depending on other params.
      validn_data, validn_size = dataset_input_fn(loaded_data, True)
      
      iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_data.output_types, train_data.output_shapes)
      features, labels = iterator.get_next()
         
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
        "rmse": tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
      }
      
      train_itr = train_data.make_one_shot_iterator()
      validn_itr = validn_data.make_initializable_iterator()
      
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

      init = tf.global_variables_initializer()

      train_steps = math.floor(train_size/BATCH_SIZE)
      # validn_steps or test steps, depending on other parameters.
      validn_steps = math.floor(validn_size/BATCH_SIZE)
      sess.run(init)
      training_handle = sess.run(train_itr.string_handle())
      validation_handle = sess.run(validn_itr.string_handle())
      best_rmse = 100000000000
      best_training_step = 0
      wait_time = 0
      for step in range(train_steps):
        try:
          sess.run(train_op, feed_dict={handle: training_handle, 
            mode: "train"})
        except tf.errors.OutOfRangeError:
          print("Reached the end of the data set.")
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
        if step % 30 == 29:
          sess.run(validn_itr.initializer)
          summ = 0
          count = 0
          for i in range(validn_steps):
            ev = sess.run(eval_metric_ops, feed_dict={handle: validation_handle,
             mode: "eval"})
            summ += ev["rmse"]
            count = i + 1
          mean_rmse = summ/count
          print("RMSE: {0:f}".format(mean_rmse))
          if mean_rmse <= best_rmse:
            best_rmse = mean_rmse
            best_training_step = step + 1
            wait_time = 0
            # Save the parameters that yield the best results.
          else:
            wait_time += 1
            if (wait_time > patience):
              print("The best training step is: {:d} \n".format(best_training_step))
              break

      #log_file.close()
      #sys.stdout = sys.__stdout__
      fold_index[0] += 1

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
