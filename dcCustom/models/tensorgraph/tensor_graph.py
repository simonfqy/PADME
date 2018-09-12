import collections
import os
import pickle
import threading
import time

import logging
import numpy as np
import tensorflow as tf
import math
import pdb
import csv
from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader
from tensorflow.python import debug as tf_debug

from dcCustom.data import NumpyDataset
from dcCustom.models.models import Model
from dcCustom.models.tensorgraph.layers import InputFifoQueue, Label, Feature, \
  Weights, Constant, Dense
from deepchem.models.tensorgraph.optimizers import Adam
from dcCustom.trans import undo_transforms
from dcCustom.utils.evaluate import GeneratorEvaluator

logger = logging.getLogger(__name__)

class TensorGraph(Model):

  def __init__(self,
               tensorboard=False,
               tensorboard_log_frequency=50,
               batch_size=100,
               random_seed=None,
               use_queue=True,
               graph=None,
               learning_rate=0.001,
               dropout_prob = 0.5,
               num_dense_layer = 3,
               dense_cmb_layer_size = 256,
               configproto=None,
               **kwargs):
    """
    Parameters
    ----------
    tensorboard: bool
      Should we log to model_dir data for tensorboard?
    tensorboard_log_frequency: int
      How many training batches before logging tensorboard?
    batch_size: int
      default batch size for training and evaluating
    use_queue: boolean
      if True when building we will create a tf.FIFO queue, which will hold
      all features, weights, and labels.  We will feed the inputs into this
      queue in batches of self.batch_size in a separate thread from the
      thread training the model.  You cannot use a queue when
      batches are not of consistent size
    graph: tensorflow.Graph
      the Graph in which to create Tensorflow objects.  If None, a new Graph
      is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    configproto: a tf.ConfigProto() object used to create tf.Session()
    """

    # Layer Management
    self.layers = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.variances = list()
    self.task_weights = list()
    self.submodels = list()
    self.loss = Constant(0)
    self.built = False
    self.queue_installed = False
    self.optimizer = Adam(
        learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7)
    self.dropout_prob = dropout_prob
    # The following two variables are for hyperparameter tuning purpose.
    self.num_dense_layer = num_dense_layer
    self.dense_cmb_layer_size = dense_cmb_layer_size
    self.configproto = configproto

    # Singular place to hold Tensor objects which don't serialize
    # These have to be reconstructed on restoring from pickle
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "FileWriter": None,
        "Graph": graph,
        "train_op": None,
        "summary_op": None,
    }
    self.tensorboard = tensorboard
    self.tensorboard_log_frequency = tensorboard_log_frequency
    self.tensorboard_step = 0
    self.global_step = 0
    self.use_queue = use_queue

    self.batch_size = batch_size
    self.random_seed = random_seed
    super(TensorGraph, self).__init__(**kwargs)
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    if self.use_queue and self.tensorboard:
      raise ValueError(
          "Currently TensorGraph cannot both use_queue and tensorboard at the same time"
      )

  def _add_layer(self, layer):
    if layer.name is None:
      layer.name = "%s_%s" % (layer.__class__.__name__, len(self.layers) + 1)
    if layer.name in self.layers:
      return
    if isinstance(layer, Feature):
      self.features.append(layer)
    if isinstance(layer, Label):
      self.labels.append(layer)
    if isinstance(layer, Weights):
      self.task_weights.append(layer)
    self.layers[layer.name] = layer
    for in_layer in layer.in_layers:
      self._add_layer(in_layer)

  def fit(self,
          dataset,
          valid_dataset=None,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          checkpoint_interval=1000,
          deterministic=False,
          restore=False,
          submodel=None,
          metric=None,
          direction=None,
          early_stopping=True,
          no_r2=False,
          evaluate_freq = 3, # Number of training epochs before evaluating
          # for early stopping.
          patience = 3,
          transformers=None,
          per_task_metrics=False,
          tasks=None,
          verbose_search=False,
          log_file=None,
          aggregated_tasks=[],
          model_name=None,
          **kwargs):
    """Train this model on a dataset.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    deterministic: bool
      if True, the samples are processed in order.  If False, a different random
      order is used for each epoch.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().
    """    
    if not early_stopping:
      return self.fit_generator(
          self.default_generator(
              dataset, epochs=nb_epoch, deterministic=deterministic),
          max_checkpoints_to_keep, checkpoint_interval, restore, submodel)
    # Using early stopping.
    
    assert evaluate_freq > 0
    assert direction is not None
    assert checkpoint_interval > 0
    assert transformers is not None
    if verbose_search:
      assert log_file is not None
    
    iterations=math.ceil(nb_epoch/evaluate_freq)
    if metric is None:
      if self.mode == 'regression':
        metric = [dcCustom.metrics.Metric(dcCustom.metrics.rms_score, np.mean,
          arithmetic_mean=True),
          dcCustom.metrics.Metric(dcCustom.metrics.concordance_index, np.mean,
          arithmetic_mean=True),
          dcCustom.metrics.Metric(dcCustom.metrics.r2_score, np.mean, arithmetic_mean=True)]
        direction = False
      elif self.mode == 'classification':
        metric = [dcCustom.metrics.Metric(dcCustom.metrics.roc_auc_score, np.mean,
          arithmetic_mean=True),
          dcCustom.metrics.Metric(dcCustom.metrics.prc_auc_score, np.mean,
          arithmetic_mean=True)]
        direction = True
    
    self.temp_model_dir = self.model_dir + "/temp"
    wait_time = 0
    
    best_score = math.inf * -1
        
    with self._get_tf("Graph").as_default():
      optimal_epoch = 0
      epoch_count = 0
      for i in range(iterations):
        num_epoch = evaluate_freq
        if i == (iterations - 1):
          num_epoch = nb_epoch - evaluate_freq * i
        # Temporarily overriding the path of saving.
        self.save_file = "%s/%s" % (self.temp_model_dir, "model")
        self.fit_generator(
            self.default_generator(
                dataset, epochs=num_epoch, deterministic=deterministic),
            max_checkpoints_to_keep, checkpoint_interval, restore, submodel)
        saver = tf.train.Saver(
              max_to_keep=max_checkpoints_to_keep, save_relative_paths=True)
        if not per_task_metrics:
          valid_score = self.evaluate(valid_dataset, metric, transformers, no_r2=no_r2,
            per_task_metrics=per_task_metrics, tasks=tasks, model_name=model_name) 
            
        else:
          valid_score, per_task_sc_val = self.evaluate(valid_dataset, metric,
            transformers, per_task_metrics=per_task_metrics, tasks=tasks, no_r2=no_r2,
            model_name=model_name)

        val_sc = 0
        per_task_sc = [0.0 for task in aggregated_tasks]  
        # we use a combination of metrics.
        if self.mode == 'regression':          
          direction = False
          for mtc in metric:
            mtc_name = mtc.metric.__name__
            composite_mtc_name = mtc.name
            if mtc_name == 'rms_score': 
              val_sc += valid_score[composite_mtc_name]
              if per_task_metrics:
                per_task_sc = [base + incr for base, incr in zip(per_task_sc, 
                  per_task_sc_val[composite_mtc_name])]

            if mtc_name == 'r2_score' or mtc_name == 'pearson_r2_score':
              if no_r2:
                continue
              val_sc += -0.5 * valid_score[composite_mtc_name]
              if per_task_metrics:
                per_task_sc = [base - 0.5 * incr for base, incr in zip(per_task_sc, 
                  per_task_sc_val[composite_mtc_name])]

            if mtc_name == 'concordance_index':
              val_sc += -valid_score[composite_mtc_name]
              if per_task_metrics:
                per_task_sc = [base - incr for base, incr in zip(per_task_sc, 
                  per_task_sc_val[composite_mtc_name])]

        elif self.mode == 'classification':          
          direction = True
          for mtc in metric:
            mtc_name = mtc.metric.__name__
            composite_mtc_name = mtc.name
            if mtc_name == 'roc_auc_score': 
              val_sc += valid_score[composite_mtc_name]
              if per_task_metrics:
                per_task_sc = [base + incr for base, incr in zip(per_task_sc, 
                  per_task_sc_val[composite_mtc_name])]

            if mtc_name == 'prc_auc_score':
              val_sc += valid_score[composite_mtc_name]
              if per_task_metrics:
                per_task_sc = [base + incr for base, incr in zip(per_task_sc, 
                  per_task_sc_val[composite_mtc_name])]

        if per_task_metrics:
          #per_task_sc_val = per_task_sc_val[metric[0].name]          
          if tasks is None:
            tasks=list(range(1, len(per_task_sc)+1))
          for task, task_sc in zip(aggregated_tasks, per_task_sc):            
            print('The score for task %s is %g' % (str(task), task_sc))
        print('The overall validation score is: ', val_sc)
        # Resuming the path of saving.
        self.save_file = "%s/%s" % (self.model_dir, "model")
        wait_time += 1
        epoch_count += num_epoch
        if not direction:
          val_sc = -1 * val_sc        
      
        if val_sc > best_score:
          best_score = val_sc
          wait_time = 0
          optimal_epoch = epoch_count
          saver.save(self.session, self.save_file, global_step=self.global_step)
          if verbose_search:
            output_val_sc = val_sc
            if not direction:
              output_val_sc = -1 * output_val_sc
            with open(log_file, 'a') as f:
              f.write("Currently at epoch number: " + str(epoch_count))
              f.write('\n')
              # Record performances
              f.write("Current best validation score: " + str(output_val_sc))
              f.write('\n')
              if per_task_metrics:         
                if tasks is None:
                  tasks=list(range(1, len(per_task_sc)+1))
                for task, task_sc in zip(aggregated_tasks, per_task_sc):            
                  f.write('The score for task %s is %g;' % (str(task), task_sc))
                f.write('\n')
        
        if (wait_time > patience):
          break       
      self.restore() #This only restores from the self.model_dir
      if not direction:
        best_score = -1 * best_score
    return [optimal_epoch, best_score]

  def fit_generator(self,
                    feed_dict_generator,
                    max_checkpoints_to_keep=5,
                    checkpoint_interval=1000,
                    restore=False,
                    submodel=None):
    """Train this model on data from a generator.

    Parameters
    ----------
    feed_dict_generator: generator
      this should generate batches, each represented as a dict that maps
      Layers to values.
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    checkpoint_interval: int
      the frequency at which to write checkpoints, measured in training steps.
      Set this to 0 to disable automatic checkpointing.
    restore: bool
      if True, restore the model from the most recent checkpoint and continue training
      from there.  If False, retrain the model from scratch.
    submodel: Submodel
      an alternate training objective to use.  This should have been created by
      calling create_submodel().

    Returns
    -------
    the average loss over the most recent checkpoint interval
    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      time1 = time.time()
      loss = self.loss
      if submodel is not None and submodel.loss is not None:
        loss = submodel.loss
      if submodel is None:
        train_op = self._get_tf('train_op')
      else:
        train_op = submodel.get_train_op()
        if submodel.loss is not None:
          loss = submodel.loss
      if checkpoint_interval > 0:
        saver = tf.train.Saver(
            max_to_keep=max_checkpoints_to_keep, save_relative_paths=True)
      if restore:
        self.restore()
      avg_loss, n_averaged_batches = 0.0, 0.0
      n_samples = 0
      n_enqueued = [0]
      final_sample = [None]
      if self.queue_installed:
        enqueue_thread = threading.Thread(
            target=_enqueue_batch,
            args=(self, feed_dict_generator, self._get_tf("Graph"),
                  self.session, n_enqueued, final_sample))
        enqueue_thread.start()
      for feed_dict in self._create_feed_dicts(feed_dict_generator, True):
        if self.queue_installed:
          # Don't let this thread get ahead of the enqueue thread, since if
          # we try to read more batches than the total number that get queued,
          # this thread will hang indefinitely.
          while n_enqueued[0] <= n_samples:
            if n_samples == final_sample[0]:
              break
            time.sleep(0)
          if n_samples == final_sample[0]:
            break
        n_samples += 1
        should_log = (self.tensorboard and
                      n_samples % self.tensorboard_log_frequency == 0)
        fetches = [train_op, loss.out_tensor]
        
        if should_log:
          fetches.append(self._get_tf("summary_op"))
        fetched_values = self.session.run(fetches, feed_dict=feed_dict)
        if should_log:
          self._log_tensorboard(fetched_values[2])
        avg_loss += fetched_values[1]
        n_averaged_batches += 1
        self.global_step += 1
        if checkpoint_interval > 0 and self.global_step % checkpoint_interval == checkpoint_interval - 1:
          saver.save(self.session, self.save_file, global_step=self.global_step)
          avg_loss = float(avg_loss) / n_averaged_batches
          logger.info('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
          avg_loss, n_averaged_batches = 0.0, 0.0
      if n_averaged_batches > 0:
        avg_loss = float(avg_loss) / n_averaged_batches
      if checkpoint_interval > 0:
        if n_averaged_batches > 0:
          logger.info('Ending global_step %d: Average loss %g' % (self.global_step,
                                                            avg_loss))
        saver.save(self.session, self.save_file, global_step=self.global_step)
        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    return avg_loss

  def _log_tensorboard(self, summary):
    """
    TODO(LESWING) set epoch
    Parameters
    ----------
    Returns
    -------
    """
    global_step = int(self.global_step)
    writer = self._get_tf("FileWriter")
    writer.reopen()
    writer.add_summary(summary, global_step=global_step)
    writer.close()

  def fit_on_batch(self, X, y, w, submodel=None):
    if not self.built:
      self.build()
    dataset = NumpyDataset(X, y)
    return self.fit(dataset, nb_epoch=1, submodel=submodel)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    if len(self.features) > 1:
      raise ValueError("More than one Feature, must use generator")
    if len(self.labels) > 1:
      raise ValueError("More than one Label, must use generator")
    if len(self.task_weights) > 1:
      raise ValueError("More than one Weights, must use generator")
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if len(self.labels) == 1 and y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b
        if len(self.features) == 1 and X_b is not None:
          feed_dict[self.features[0]] = X_b
        if len(self.task_weights) == 1 and w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        for (initial_state, zero_state) in zip(self.rnn_initial_states,
                                               self.rnn_zero_states):
          feed_dict[initial_state] = zero_state
        
        yield feed_dict

  def _predict(self, generator, transformers, outputs, uncertainty):
    """
    Predict outputs for data provided by a generator.

    This is the private implementation of prediction. Do not call it directly.
    Instead, call one of the public prediction methods.

    Parameters
    ----------
    generator: Generator
      Generator that constructs feed dictionaries for TensorGraph.
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs.
      If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.
    uncertainty: bool
      specifies whether this is being called as part of estimating uncertainty.
      If True, it sets the training flag so that dropout will be enabled, and 
      returns the values of the uncertainty outputs.
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    if outputs is None:
      outputs = self.outputs
    elif not isinstance(outputs, collections.Sequence):
      outputs = [outputs]
    if uncertainty:
      if len(self.variances) == 0:
        raise ValueError('This model cannot compute uncertainties.')
      if len(self.variances) != len(outputs):
        raise ValueError(
          'The number of variances must exactly match the number of outputs.')
      tensors = outputs + self.variances
    else:
      tensors = outputs

    with self._get_tf("Graph").as_default():
      # Gather results for each output
      results = [[] for out in tensors]
      n_samples = 0
      n_enqueued = [0]
      final_sample = [None]
      if self.queue_installed:
        enqueue_thread = threading.Thread(
            target=_enqueue_batch,
            args=(self, generator, self._get_tf("Graph"), self.session,
                  n_enqueued, final_sample))
        enqueue_thread.start()
      # batch_ind = 1
      for feed_dict in self._create_feed_dicts(generator, False):
        if self.queue_installed:
          # Don't let this thread get ahead of the enqueue thread, since if
          # we try to read more batches than the total number that get queued,
          # this thread will hang indefinitely.
          while n_enqueued[0] <= n_samples:
            if n_samples == final_sample[0]:
              break
            time.sleep(0)
          if n_samples == final_sample[0]:
            break
        n_samples += 1
        feed_results = self.session.run(tensors, feed_dict=feed_dict)
        # if batch_ind >= 84:
        #   pdb.set_trace()
        if len(feed_results) > 1:
          if len(transformers):
            raise ValueError("Does not support transformations "
                             "for multiple outputs.")
        elif len(feed_results) == 1:
          result = undo_transforms(feed_results[0], transformers)
          feed_results = [result]
        for ind, result in enumerate(feed_results):
          results[ind].append(result)
        # batch_ind += 1
      #pdb.set_trace()
      final_results = []
      for result_list in results:
        final_results.append(np.concatenate(result_list, axis=0))
      # pdb.set_trace()
      # If only one output, just return array
      if len(final_results) == 1:
        return final_results[0]
      elif uncertainty:
        return zip(final_results[:len(outputs)], final_results[len(outputs):])
      else:
        return final_results

  def predict_on_generator(self, generator, transformers=[], outputs=None):
    """
    Parameters
    ----------
    generator: Generator
      Generator that constructs feed dictionaries for TensorGraph.
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs.
      If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    return self._predict(generator, transformers, outputs, False)

  def predict_on_batch(self, X, transformers=[], outputs=None):
    """Generates predictions for input samples, processing samples in a batch.

    Parameters
    ----------
    X: ndarray
      the input data, as a Numpy array.
    transformers: List
      List of dc.trans.Transformers

    Returns
    -------
    A Numpy array of predictions.
    """
    dataset = NumpyDataset(X=X, y=None)
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers, outputs)

  def predict_uncertainty_on_batch(self, X, masks=50):
    dataset = NumpyDataset(X=X, y=None)
    return self.predict_uncertainty(dataset, masks)
  
  @staticmethod
  def output_predictions(n_tasks, dataset, y_preds, csv_out):
    """
    Writes predictions to file.

    Args:
      y_preds: np.ndarray
      csv_out: CSV file name.
    """
    mol_ids = dataset.ids
    proteins = dataset.X[:, 1]    
    y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
    #pdb.set_trace()
    assert len(y_preds) == len(mol_ids)
    with open(csv_out, "w") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["Compound", "proteinName", "protein_dataset"] + dataset.get_task_names())
      for i in range(len(y_preds)):
        mol_id = mol_ids[i]
        y_pred = y_preds[i]
        protein = proteins[i]
        prot_source_and_name = protein.get_name()
        prot_name = prot_source_and_name[1]
        prot_dataset = prot_source_and_name[0]
        csvwriter.writerow([mol_id, prot_name, prot_dataset] + list(y_pred))

  def predict(self, dataset, transformers=[], outputs=None, csv_out=None, tasks=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.
    outputs: object
      If outputs is None, then will assume outputs = self.outputs. If outputs is 
      a Layer/Tensor, then will evaluate and return as a single ndarray. If 
      outputs is a list of Layers/Tensors, will return a list of ndarrays.

    Returns
    -------
    results: numpy ndarray or list of numpy ndarrays
    """
    generator = self.default_generator(dataset, predict=True, pad_batches=False) 
    if csv_out is not None:
      self.restore()
      predictions = self.predict_on_generator(generator, transformers, outputs)
      self.output_predictions(len(tasks), dataset, predictions, csv_out)
    else:
      predictions = self.predict_on_generator(generator, transformers, outputs)
    return predictions

  # TODO: I left it as a stub. Need to populate it if required.
  def predict_uncertainty(self, dataset, masks=50):
    return None

  def topsort(self):

    def add_layers_to_list(layer, sorted_layers):
      if layer in sorted_layers:
        return
      for in_layer in layer.in_layers:
        add_layers_to_list(in_layer, sorted_layers)
      sorted_layers.append(layer)

    sorted_layers = []
    for l in self.features + self.labels + self.task_weights + self.outputs + self.variances:
      add_layers_to_list(l, sorted_layers)
    add_layers_to_list(self.loss, sorted_layers)
    for submodel in self.submodels:
      if submodel.loss is not None:
        add_layers_to_list(submodel.loss, sorted_layers)
    return sorted_layers

  def build(self):
    if self.built:
      return
    with self._get_tf("Graph").as_default():
      self._training_placeholder = tf.placeholder(dtype=tf.float32, shape=())
      if self.random_seed is not None:
        tf.set_random_seed(self.random_seed)
      self._install_queue()
      for layer in self.topsort():
        with tf.name_scope(layer.name):
          layer.create_tensor(training=self._training_placeholder)
          self.rnn_initial_states += layer.rnn_initial_states
          self.rnn_final_states += layer.rnn_final_states
          self.rnn_zero_states += layer.rnn_zero_states
          if self.tensorboard:
            if type(layer) is Dense:
              tf.summary.scalar('%s/fraction_of_zero_values'%layer.name, 
                tf.nn.zero_fraction(layer.out_tensor)) 
              layer.set_summary('histogram')
              layer.add_summary_to_tg()
      self.session = tf.Session(config=self.configproto)
      #self.session = tf_debug.LocalCLIDebugWrapperSession(self.session, dump_root = './tfdbg')
      #self.session.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
      self.built = True

      # Ensure all training operators have been created.

      self._get_tf('train_op')
      for submodel in self.submodels:
        train_op = submodel.get_train_op()

      # Initialize variables.
      self.session.run(tf.global_variables_initializer())
      for layer in self.layers.values():
        if layer.variable_values is not None:
          variables = self.get_layer_variables(layer)
          for var, val in zip(variables, layer.variable_values):
            self.session.run(var.assign(val))

    for layer in self.layers.values():
      if layer.tensorboard:
        self.tensorboard = True
    tf.summary.scalar("loss", self.loss.out_tensor)
    tf.summary.scalar("avg_loss", self.loss.out_tensor/self.batch_size)
    for layer in self.layers.values():
      if layer.tensorboard:
        tf.summary.tensor_summary(layer.name, layer.out_tensor)
    if self.tensorboard:
      writer = self._get_tf("FileWriter")
      writer.add_graph(self._get_tf("Graph"))
      writer.close()

    # As a sanity check, make sure all tensors have the correct shape.

    for layer in self.layers.values():
      try:
        assert list(layer.shape) == layer.out_tensor.get_shape().as_list(
        ), '%s: Expected shape %s does not match actual shape %s' % (
            layer.name, layer.shape, layer.out_tensor.get_shape().as_list())
      except NotImplementedError:
        pass

  def _install_queue(self):
    """
    """
    if not self.use_queue or self.queue_installed:
      for layer in self.features + self.labels + self.task_weights:
        layer.pre_queue = True
      return
    inputs = self.features + self.labels + self.task_weights
    if len(inputs) == 0:
      return
    names = []
    shapes = []
    pre_q_inputs = []
    q = InputFifoQueue(shapes, names, in_layers=pre_q_inputs)
    q.name = "%s_%s" % (q.__class__.__name__, len(self.layers) + 1)

    for layer in inputs:
      pre_q_input = layer.create_pre_q()
      shapes.append(pre_q_input.shape)
      names.append(pre_q_input.name)
      pre_q_inputs.append(pre_q_input)

      layer.in_layers.append(q)

    self._add_layer(q)
    self.input_queue = q
    self.queue_installed = True

  def set_loss(self, layer):
    self._add_layer(layer)
    self.loss = layer

  def add_output(self, layer):
    '''Add an output layer that can be computed by predict()'''
    self._add_layer(layer)
    self.outputs.append(layer)

  def add_variance(self, layer):
    """Add a layer that computes the variance in an output.

    If a model supports uncertainty, it must call add_variance() once for every
    output. Each variance layer has the same shape as the corresponding output,
    and each element computes an estimate of the variance from aleatoric
    uncertainty in the corresponding element of the output.

    In addition, if a model supports uncertainty it MUST use dropout on every
    layer. Otherwise, the uncertainties it computes will be inaccurate.

    """
    self._add_layer(layer)
    self.variances.append(layer)

  def set_optimizer(self, optimizer):
    """Set the optimizer to use for fitting."""
    self.optimizer = optimizer

  def create_submodel(self, layers=None, loss=None, optimizer=None):
    """Create an alternate objective for training one piece of a TensorGraph.

    A TensorGraph consists of a set of layers, and specifies a loss function and
    optimizer to use for training those layers.  Usually this is sufficient, but
    there are cases where you want to train different parts of a model separately.
    For example, a GAN consists of a generator and a discriminator.  They are
    trained separately, and they use different loss functions.

    A submodel defines an alternate objective to use in cases like this.  It may
    optionally specify any of the following: a subset of layers in the model to
    train; a different loss function; and a different optimizer to use.  This
    method creates a submodel, which you can then pass to fit() to use it for
    training.

    Parameters
    ----------
    layers: list
      the list of layers to train.  If None, all layers in the model will be
      trained.
    loss: Layer
      the loss function to optimize.  If None, the model's main loss function
      will be used.
    optimizer: Optimizer
      the optimizer to use for training.  If None, the model's main optimizer
      will be used.

    Returns
    -------
    the newly created submodel, which can be passed to any of the fitting
    methods.
    """
    if self.built:
      raise ValueError('Submodels must be created before build() is called.')
    submodel = Submodel(self, layers, loss, optimizer)
    self.submodels.append(submodel)
    if loss is not None:
      self._add_layer(loss)
    return submodel

  def get_pickling_errors(self, obj, seen=None):
    if seen == None:
      seen = []
    try:
      state = obj.__getstate__()
    except AttributeError:
      return
    if state == None:
      return
    if isinstance(state, tuple):
      if not isinstance(state[0], dict):
        state = state[1]
      else:
        state = state[0].update(state[1])
    result = {}
    for i in state:
      try:
        pickle.dumps(state[i], protocol=2)
      except pickle.PicklingError:
        if not state[i] in seen:
          seen.append(state[i])
          result[i] = self.get_pickling_errors(state[i], seen)
    return result

  def save(self):
    # Remove out_tensor from the object to be pickled
    must_restore = False
    tensor_objects = self.tensor_objects
    rnn_initial_states = self.rnn_initial_states
    rnn_final_states = self.rnn_final_states
    rnn_zero_states = self.rnn_zero_states
    session = self.session
    self.tensor_objects = {}
    self.rnn_initial_states = []
    self.rnn_final_states = []
    self.rnn_zero_states = []
    self.session = None
    out_tensors = []
    submodel_ops = []
    if self.built:
      must_restore = True
      for layer in self.topsort():
        out_tensors.append(layer.none_tensors())
      for submodel in self.submodels:
        submodel_ops.append(submodel._train_op)
        submodel._train_op = None
      training_placeholder = self._training_placeholder
      self._training_placeholder = None
      self.built = False

    # Pickle itself
    pickle_name = os.path.join(self.model_dir, "model.pickle")

    with open(pickle_name, 'wb') as fout:
      try:
        pickle.dump(self, fout)
      except Exception as e:
        logger.info(self.get_pickling_errors(self))
        raise e

    # add out_tensor back to everyone
    if must_restore:
      for index, layer in enumerate(self.topsort()):
        layer.set_tensors(out_tensors[index])
      for submodel, op in zip(self.submodels, submodel_ops):
        submodel._train_op = op
      self._training_placeholder = training_placeholder
      self.built = True
    self.tensor_objects = tensor_objects
    self.rnn_initial_states = rnn_initial_states
    self.rnn_final_states = rnn_final_states
    self.rnn_zero_states = rnn_zero_states
    self.session = session

  def evaluate_generator(self,
                         feed_dict_generator,
                         metrics,
                         transformers=[],
                         dataset=None,
                         labels=None,
                         outputs=None,
                         weights=[],
                         per_task_metrics=False,
                         no_r2=False,
                         no_concordance_index=False,
                         plot=False,
                         is_training_set=False,
                         tasks=None,
                         model_name=None):

    if labels is None:
      raise ValueError
    n_tasks = len(self.outputs)
    if tasks is not None:
      assert len(tasks) == n_tasks
    n_classes = self.outputs[0].out_tensor.get_shape()[-1].value
    evaluator = GeneratorEvaluator(
        self,
        feed_dict_generator,
        transformers,
        dataset=dataset,
        labels=labels,
        outputs=outputs,
        weights=weights,
        n_tasks=n_tasks,
        n_classes=n_classes,
        is_training_set=is_training_set,
        tasks=tasks,
        model_name=model_name)
    if not per_task_metrics:
      scores = evaluator.compute_model_performance(metrics, 
        no_concordance_index=no_concordance_index, plot=plot, no_r2=no_r2)
      return scores
    else:
      scores, per_task_scores = evaluator.compute_model_performance(
        metrics, per_task_metrics=per_task_metrics, 
        no_concordance_index=no_concordance_index, plot=plot, no_r2=no_r2)
      return scores, per_task_scores

  def get_layer_variables(self, layer):
    """Get the list of trainable variables in a layer of the graph."""
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      if layer.variable_scope == '':
        return []
      return tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=layer.variable_scope)

  def get_variables(self):
    """Get the list of all trainable variables in the graph."""
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  def get_global_step(self):
    return self._get_tf("GlobalStep")

  def _get_tf(self, obj):
    """Fetches underlying TensorFlow primitives.

    Parameters
    ----------
    obj: str
      If "Graph", returns tf.Graph instance. If "FileWriter", returns
      tf.summary.FileWriter. If "Optimizer", returns the optimizer. If
      "train_op", returns the train operation. If "summary_op", returns the
      merged summary. If "GlobalStep" returns the global step.
    Returns
    -------
    TensorFlow Object

    """

    if obj in self.tensor_objects and self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects['Graph'] = tf.Graph()
    elif obj == "FileWriter":
      self.tensor_objects['FileWriter'] = tf.summary.FileWriter(self.model_dir)
    elif obj == 'Optimizer':
      self.tensor_objects['Optimizer'] = self.optimizer._create_optimizer(
          self._get_tf('GlobalStep'))
    elif obj == 'train_op':
      opt = self._get_tf('Optimizer')
      global_step = self._get_tf('GlobalStep')
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        try:
          self.tensor_objects['train_op'] = opt.minimize(
              self.loss.out_tensor, global_step=global_step)
        except ValueError:
          # The loss doesn't depend on any variables.
          self.tensor_objects['train_op'] = 0
    elif obj == 'summary_op':
      self.tensor_objects['summary_op'] = tf.summary.merge_all(
          key=tf.GraphKeys.SUMMARIES)
    elif obj == 'GlobalStep':
      with self._get_tf("Graph").as_default():
        self.tensor_objects['GlobalStep'] = tf.Variable(0, trainable=False)
    return self._get_tf(obj)

  def save_checkpoint(self, max_checkpoints_to_keep=5):
    """Save a checkpoint to disk.

    Usually you do not need to call this method, since fit() saves checkpoints
    automatically.  If you have disabled automatic checkpointing during fitting,
    this can be called to manually write checkpoints.

    Parameters
    ----------
    max_checkpoints_to_keep: int
      the maximum number of checkpoints to keep.  Older checkpoints are discarded.
    """
    saver = tf.train.Saver(
      self.get_variables(), max_to_keep=max_checkpoints_to_keep)
    saver.save(self.session, self.save_file, global_step=self.global_step)

  def get_checkpoints(self):
    """Get a list of all available checkpoint files."""
    return tf.train.get_checkpoint_state(
        self.model_dir).all_model_checkpoint_paths

  def restore(self, checkpoint=None):
    """Reload the values of all variables from a checkpoint file.

    Parameters
    ----------
    checkpoint: str
      the path to the checkpoint file to load.  If this is None, the most recent
      checkpoint will be chosen automatically.  Call get_checkpoints() to get a
      list of all available checkpoints.
    """
    if not self.built:
      self.build()
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint is None:
      raise ValueError('No checkpoint found')
    with self._get_tf("Graph").as_default():
      reader = NewCheckpointReader(checkpoint)
      var_names = set([x for x in reader.get_variable_to_shape_map()])
      # var_map = {
      #     x.op.name: x
      #     for x in tf.global_variables()
      #     if x.op.name in var_names
      # }
      # saver = tf.train.Saver(var_list=var_map)
      var_list = []
      for var in self.get_variables():
        name = var.name
        if ':' in name:
          name = name[:name.rfind(':')]
        if name in var_names:
          var_list.append(var)
      saver = tf.train.Saver(var_list=var_list)
      saver.restore(self.session, checkpoint)

  def get_num_tasks(self):
    return len(self.outputs)

  def get_pre_q_input(self, input_layer):
    layer_name = input_layer.name
    pre_q_name = "%s_pre_q" % layer_name
    return self.layers[pre_q_name]

  @staticmethod
  def load_from_dir(model_dir, restore=True):
    pickle_name = os.path.join(model_dir, "model.pickle")
    with open(pickle_name, 'rb') as fout:
      tensorgraph = pickle.load(fout)
      tensorgraph.built = False
      tensorgraph.model_dir = model_dir
      if restore:
        try:
          tensorgraph.restore()
        except ValueError:
          pass  # No checkpoint to load
      return tensorgraph

  def __del__(self):
    pass

  def _create_feed_dicts(self, generator, training):
    """Create feed dicts for use in fitting or prediction.

    Parameters
    ----------
    generator: Generator
      the feed dict generator that was passed to fit_generator() or predict_on_generator()
    training: bool
      True during training, False during prediction
    """
    train_value = 1.0 if training else 0.0
    if self.queue_installed:
      while True:
        yield {self._training_placeholder: train_value}
    for d in generator:
      feed_dict = dict(d)
      feed_dict[self._training_placeholder] = train_value
      yield feed_dict

  def make_estimator(self, feature_columns, weight_column=None, metrics={},
    model_dir=None, config=None):
    """Construct a Tensorflow Estimator from this model.

    tf.estimator.Estimator is the standard Tensorflow API for representing models.
    This method provides interoperability between DeepChem and other Tensorflow
    based tools by allowing any model to be used in an Estimator.

    Once this method returns, the Estimator it created is independent of the model
    it was created from. They do not share tensors, variables, save files, or any
    other resources. The Estimator is a self contained object with its own methods
    for training, evaluation, prediction, checkpointing, etc.

    Parameters
    ----------
    feature_columns: list of tf.feature_column objects
      this describes the input features to the models. There must be one entry for
      each Feature layer in this model's features field.
    weight_column: tf.feature_column or None
      if this model includes a Weights layer, this describes the input weights.
      Otherwise, this should be None.
    metrics: map
      metrics that should be computed in calls to evaluate(). For each entry,
      the key is the name to report for the metric, and the value is a function
      of the form f(labels, predictions, weights) that returns the tensors for
      computing the metric. Any of the functions in tf.metrics can be used, as
      can other functions that satisfy the same interface.
    model_dir: str
      the directory in which the Estimator should save files. If None, this
      defaults to the model's model_dir.
    config: RunConfig
      configuration options for the Estimator.
    """

    if len(feature_columns) != len(self.features):
      raise ValueError(
        'This model requries %d feature column(s)' % len(self.features))
    if len(self.labels) != 1:
      raise ValueError(
        'Can only create an Estimator from a model with exactly one Label input'
      )
    if len(self.task_weights) > 1:
      raise ValueError(
        'Cannot create an Estimator from a model with multiple Weight inputs')
    if weight_column is None:
      if len(self.task_weights) > 0:
        raise ValueError(
          'Cannot specify weight_column for a model with no Weight inputs')
    if model_dir is None:
      model_dir = self.model_dir

    # Define a function that recursively creates tensors from layers.

    def create_tensors(layer, tensors, training):
      if layer in tensors:
        return tensors[layer]
      inputs = [
        create_tensors(in_layer, tensors, training)
        for in_layer in layer.in_layers
      ]
      tensor = layer.create_tensor(
        in_layers=inputs, set_tensors=False, training=training)
      tensors[layer] = tensor
      layer.add_summary_to_tg(tensor)
      return tensor

    # Define the model function.
    def model_fn(features, labels, mode):
      # Define the inputs.
      tensors = self.create_estimator_inputs(feature_columns, weight_column,
        features, labels, mode)
      for layer, tensor in tensors.items():
        layer.add_summary_to_tg(tensor)

      # Create the correct outputs, based on the mode.
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        for i, output in enumerate(self.outputs):
          predictions[i] = create_tensors(output, tensors, 0)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      if mode == tf.estimator.ModeKeys.EVAL:
        loss = create_tensors(self.loss, tensors, 0)
        predictions = create_tensors(self.outputs[0], tensors, 0)
        if len(self.task_weights) == 0:
          weights = None
        else:
          weights = tensors[self.task_weights[0]]
        eval_metric_ops = {}
        for name, function in metrics.items():
          eval_metric_ops[name] = function(tensors[self.labels[0]], predictions, weights)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
      if mode == tf.estimator.ModeKeys.TRAIN:
        loss = create_tensors(self.loss, tensors, 1)
        global_step = tf.train.get_global_step()
        optimizer = self.optimizer._create_optimizer(global_step)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
      raise ValueError('Unknown mode')

    return tf.estimator.Estimator(
      model_fn=model_fn, model_dir=model_dir, config=config)

  def create_estimator_inputs(self, feature_columns, weight_column, features,
    labels, mode):
    """This is called by make_estimator() to create tensors for the inputs.

    feature_columns and weight_column are the arguments passed to make_estimator().
    Features, labels and mode are the arguments passed to the estimator's
    model function. THis method creates and returns a dict with one entry for
    every Feature, Label, or Weight layer in the graph. The keys are the layers,
    and the values are the tensors that correspond to them.

    Any subclass that overrides default_generator() must also override this
    method.
    """
    if self.__class__.default_generator != TensorGraph.default_generator:
      raise ValueError(
        "Class overrides default_generator() but not create_estimator_inputs()"
      )
    tensors = {}
    for layer, column in zip(self.features, feature_columns):
      tensors[layer] = tf.feature_column.input_layer(features, [column])
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
        features, [weight_column])
    if labels is not None:
      tensors[self.labels[0]] = tf.cast(labels, self.labels[0].dtype)
    return tensors

def _enqueue_batch(tg, generator, graph, sess, n_enqueued, final_sample):
  """
  Function to load data into
  Parameters
  ----------
  tg
  dataset
  graph
  sess

  Returns
  -------

  """
  with graph.as_default():
    num_samples = 0
    for feed_dict in generator:
      enq = {}
      enq[tg._training_placeholder] = 1.0
      for layer in tg.features + tg.labels + tg.task_weights:
        if layer in feed_dict:
          value = feed_dict[layer]
          # Add or remove dimensions of size 1 to match the shape of the layer.
          try:
            value_dims = len(value.shape)
            layer_dims = len(layer.shape)
            if value_dims < layer_dims:
              if all(i == 1 for i in layer.shape[value_dims:]):
                value = value.reshape(
                  list(value.shape) + [1] * (layer_dims - value_dims))
            if value_dims > layer_dims:
              if all(i == 1 for i in value.shape[layer_dims:]):
                value = value.reshape(value.shape[:layer_dims])
          except:
            pass
        else:
          value = np.zeros(
              [0] + list(layer.shape[1:]), dtype=layer.dtype.as_numpy_dtype)
        enq[tg.get_pre_q_input(layer).out_tensor] = value
      sess.run(tg.input_queue.out_tensor, feed_dict=enq)
      n_enqueued[0] += 1
    final_sample[0] = n_enqueued[0]


class TFWrapper(object):
  """This class exists as a workaround for Tensorflow objects not being picklable.

  The job of a TFWrapper is to create Tensorflow objects by passing defined arguments
  to a constructor.  There are cases where we really want to store Tensorflow objects
  of various sorts (optimizers, initializers, etc.), but we can't because they cannot
  be pickled.  So instead we store a TFWrapper that creates the object when needed.
  """

  def __init__(self, tf_class, **kwargs):
    """Create a TFWrapper for constructing a Tensorflow object.

    Parameters
    ----------
    tf_class: class
      the type of object to create
    kwargs:
      any other arguments will be passed on to the object's constructor
    """
    self.tf_class = tf_class
    self.kwargs = kwargs

  def __call__(self):
    return self.tf_class(**self.kwargs)


class Submodel(object):
  """An alternate objective for training one piece of a TensorGraph."""

  def __init__(self, graph, layers, loss, optimizer):
    """Create a submodel.

    In normal use, you should call create_submodel() on the TensorGraph instead
    of using this constructor directly."""
    self.graph = graph
    self.layers = layers
    self.loss = loss
    self.optimizer = optimizer
    self._train_op = None

  def get_train_op(self):
    """Get the Tensorflow operator to use for training."""
    if self._train_op is None:
      if self.layers is None:
        variables = None
      else:
        variables = []
        for layer in self.layers:
          variables += self.graph.get_layer_variables(layer)
      if self.loss is None:
        loss = self.graph.loss
      else:
        loss = self.loss
      # if self.optimizer is None:
      #   optimizer = self.graph.optimizer
      # else:
      #   optimizer = self.optimizer
      tf_opt = self.create_optimizer()
      global_step = self.graph._get_tf('GlobalStep')
      self._train_op = tf_opt.minimize(loss.out_tensor, global_step, variables)
    return self._train_op

  def create_optimizer(self):
    """Create the Tensorflow optimizer to use for training."""
    if self.optimizer is None:
      optimizer = self.graph.optimizer
    else:
      optimizer = self.optimizer
    global_step = self.graph._get_tf('GlobalStep')
    return optimizer._create_optimizer(global_step)