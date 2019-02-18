"""Evaluation metrics."""

import numpy as np
import pdb
import time
import warnings
import os
import re
import copy
import itertools
from dcCustom.metrics.cindex_measure import cindex
from multiprocessing import Pool
from time import gmtime, strftime
from deepchem.utils.save import log
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import OrderedDict


def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)


def compute_roc_auc_scores(y, y_pred):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  try:
    score = roc_auc_score(y, y_pred)
  except ValueError:
    warnings.warn("ROC AUC score calculation failed.")
    score = 0.5
  return score


def balanced_accuracy_score(y, y_pred):
  """Computes balanced accuracy score."""
  num_positive = float(np.count_nonzero(y))
  num_negative = float(len(y) - num_positive)
  pos_weight = num_negative / num_positive
  weights = np.ones_like(y)
  weights[y != 0] = pos_weight
  return accuracy_score(y, y_pred, sample_weight=weights)


def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2


def prc_auc_score(y, y_pred):
  """Compute area under precision-recall curve"""
  assert y_pred.shape == y.shape
  assert y_pred.shape[1] == 2
  precision, recall, _ = precision_recall_curve(y[:, 1], y_pred[:, 1])
  return auc(recall, precision)


def rms_score(y_true, y_pred):
  """Computes RMS error."""
  return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
  """Computes MAE."""
  return mean_absolute_error(y_true, y_pred)

def concordance_index(y_true, y_pred):
  return cindex(y_true, y_pred)

def kappa_score(y_true, y_pred):
  """Calculate Cohen's kappa for classification tasks.

  See https://en.wikipedia.org/wiki/Cohen%27s_kappa

  Note that this implementation of Cohen's kappa expects binary labels.

  Args:
    y_true: Numpy array containing true values.
    y_pred: Numpy array containing predicted values.

  Returns:
    kappa: Numpy array containing kappa for each classification task.

  Raises:
    AssertionError: If y_true and y_pred are not the same size, or if class
      labels are not in [0, 1].
  """
  assert len(y_true) == len(y_pred), 'Number of examples does not match.'
  yt = np.asarray(y_true, dtype=int)
  yp = np.asarray(y_pred, dtype=int)
  assert np.array_equal(
      np.unique(yt), [0,
                      1]), ('Class labels must be binary: %s' % np.unique(yt))
  observed_agreement = np.true_divide(
      np.count_nonzero(np.equal(yt, yp)), len(yt))
  expected_agreement = np.true_divide(
      np.count_nonzero(yt == 1) * np.count_nonzero(yp == 1) +
      np.count_nonzero(yt == 0) * np.count_nonzero(yp == 0), len(yt)**2)
  kappa = np.true_divide(observed_agreement - expected_agreement,
                         1.0 - expected_agreement)
  return kappa


class Metric(object):
  """Wrapper class for computing user-defined metrics."""

  def __init__(self,
               metric,
               task_averager=None,
               arithmetic_mean=False,
               name=None,
               threshold=None,
               verbose=True,
               mode=None,
               compute_energy_metric=False,
               aggregate_list=[],
               weighted_metric_of_each_endpoint=False):
    """
    Args:
      metric: function that takes args y_true, y_pred (in that order) and
              computes desired score.
      task_averager: If not None, should be a function that averages metrics
              across tasks. For example, task_averager=np.mean. If task_averager
              is provided, this task will be inherited as a multitask metric.
    """
    self.metric = metric
    self.task_averager = task_averager
    self.arithmetic_mean = arithmetic_mean
    self.is_multitask = (self.task_averager is not None)
    if name is None:
      if not self.is_multitask:
        self.name = self.metric.__name__
      else:
        self.name = self.task_averager.__name__ + "-" + self.metric.__name__
    else:
      self.name = name
    self.verbose = verbose
    self.threshold = threshold
    if mode is None:
      if self.metric.__name__ in [
          "roc_auc_score", "matthews_corrcoef", "recall_score",
          "accuracy_score", "kappa_score", "precision_score",
          "balanced_accuracy_score", "prc_auc_score"
      ]:
        mode = "classification"
      elif self.metric.__name__ in [
          "pearson_r2_score", "r2_score", "mean_squared_error",
          "mean_absolute_error", "rms_score", "mae_score", "concordance_index"
      ]:
        mode = "regression"
      else:
        raise ValueError("Must specify mode for new metric.")
    assert mode in ["classification", "regression"]
    self.mode = mode
    self.aggregate_list = aggregate_list
    self.weighted_metric_of_each_endpoint = weighted_metric_of_each_endpoint
    # The convention used is that the first task is the metric.
    # TODO(rbharath, joegomes): This doesn't seem like it should be hard-coded as
    # an option in the Metric class. Instead, this should be possible to move into
    # user-space as a custom task_averager function.
    self.compute_energy_metric = compute_energy_metric

  def get_y_vectors(self, y_true, y_pred, w):
    """Get the y vectors

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.

    Returns:
      Two vectors corresponding to true and predicted values of y, respectively.

    Raises:
      
    """  
    y_true = np.array(np.squeeze(y_true[w != 0]))
    y_pred = np.array(np.squeeze(y_pred[w != 0]))

    if len(y_true.shape) == 0:
      n_samples = 1
    else:
      n_samples = y_true.shape[0]
    # If there are no nonzero samples, metric is ill-defined.
    if not y_true.size:
      return np.nan, np.nan

    y_true = np.reshape(y_true, (n_samples,))
    if self.mode == "classification":
      n_classes = y_pred.shape[-1]
      # TODO(rbharath): This has been a major source of bugs. Is there a more
      # robust characterization of which metrics require class-probs and which
      # don't?
      if "roc_auc_score" in self.name or "prc_auc_score" in self.name:
        y_true = to_one_hot(y_true).astype(int)
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
      else:
        y_true = y_true.astype(int)
        # Reshape to handle 1-d edge cases
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
        y_pred = from_one_hot(y_pred)
    else:
      y_pred = np.reshape(y_pred, (n_samples,))

    if self.threshold is not None:
      y_pred = np.greater(y_pred, self.threshold) * 1
      y_true = np.greater(y_true, self.threshold) * 1
      n_classes = 2
      y_pred = to_one_hot(y_pred).astype(int)
      y_true = to_one_hot(y_true).astype(int) 

    return y_true, y_pred
    
    
  def get_metric_values_for_tasks(self, excluded_single_tasks_dict, computed_metrics, tasks, y_true, y_pred,
    w, plot, taskind_to_metatask=None, num_observations=None, all_metrics=None, is_training_set=None,
    no_concordance_index=False, model_name=None, do_aggregation=False, task_name_to_metric_value=None):  
    
    for task, task_name in enumerate(tasks):
      if do_aggregation: 
        if taskind_to_metatask and task in taskind_to_metatask:
          continue
      y_task = y_true[:, task]
      if self.mode == "regression":
        y_pred_task = y_pred[:, task]
      else:
        y_pred_task = y_pred[:, task, :]
      w_task = w[:, task]
      
      whether_plot = plot
      if task in taskind_to_metatask:
        whether_plot = False

      calculate_metric = True      
      if sum(w_task != 0) > 0:        
        if self.metric.__name__ in {'concordance_index', 'r2_score'}:
          # See whether the y_true vector has more than 0 valid pairs. If so, calculate. 
          y_true_task_vec, _ = self.get_y_vectors(y_task, y_pred_task, w_task)
          unequal_count = sum(y_true_task_vec[0] != y_true_task_vec)
          if unequal_count <= 0:
            calculate_metric = False
            excluded_single_tasks_dict[task] = num_observations[task]
            
        if calculate_metric:
          metric_value = self.compute_singletask_metric(y_task, y_pred_task, w_task, 
            plot=whether_plot, all_metrics=all_metrics, is_training_set=is_training_set, 
            no_concordance_index=no_concordance_index, task_name=task_name, 
            model_name=model_name)        
      else:
        calculate_metric = False

      if not calculate_metric:
        metric_value = np.nan
      computed_metrics.append(metric_value)
      if task_name_to_metric_value is not None:
        assert task_name not in task_name_to_metric_value
        task_name_to_metric_value[task_name] = metric_value
      
      
  def plot_and_get_metric_values_for_metatasks(self, meta_task_list, metatask_to_task, computed_metrics,
    y_true, y_pred, w, plot=False, all_metrics=None, is_training_set=False, no_concordance_index=False, 
    model_name=None, excluded_metatasks_dict=None, aggregated_num_obs=None, 
    task_name_to_metric_value=None, num_observations=None):    
    
    task_names_so_far = set()
    for meta_task_name in meta_task_list:
      if plot:
        min_list = []
        max_list = []
      y_pred_meta = np.array([])
      y_true_meta = np.array([])     
      if self.weighted_metric_of_each_endpoint:
        single_task_name_to_num_obs = OrderedDict()
        single_task_name_to_metric_value = OrderedDict()
      
      for pair in metatask_to_task[meta_task_name]:
        task_ind = pair[0]
        task_name = pair[1]
        assert task_name not in task_names_so_far
        task_names_so_far.add(task_name)
        y_task = y_true[:, task_ind]
        if self.mode == "regression":
          y_pred_task = y_pred[:, task_ind]
        else:
          y_pred_task = y_pred[:, task_ind, :]
        w_task = w[:, task_ind]
        
        all_missing = sum(w_task != 0) <= 0
        
        if self.weighted_metric_of_each_endpoint:
          assert self.arithmetic_mean
          assert num_observations is not None
          # TODO: this whole thing should be modularized as one function, until the elif.
          # Calculate the metric for each endpoint.
          
          calculate_metric = True
          if all_missing:     
            calculate_metric = False
          else:
            if self.metric.__name__ in {'concordance_index', 'r2_score'}:
              # See whether the y_true vector has more than 0 valid pairs. If so, calculate. 
              y_true_task_vec, _ = self.get_y_vectors(y_task, y_pred_task, w_task)
              calculate_metric = (sum(y_true_task_vec[0] != y_true_task_vec) >= 1)
                                            
            if calculate_metric:
              # When self.weighted_metric_of_each_endpoint and plot is True, then plotting is done for 
              # each endpoint individually and for the whole aggregated task.
              metric_value_for_single_task = self.compute_singletask_metric(y_task, y_pred_task, w_task, 
                plot=plot, all_metrics=all_metrics, is_training_set=is_training_set, 
                no_concordance_index=no_concordance_index, task_name=task_name, model_name=model_name)      
                               
          if not calculate_metric:
            metric_value_for_single_task = np.nan
          
          single_task_name_to_num_obs[task_name] = num_observations[task_ind]
          single_task_name_to_metric_value[task_name] = metric_value_for_single_task
        
        if all_missing:
          continue       
                        
        y_true_task, y_pred_task = self.get_y_vectors(y_task, y_pred_task, w_task)          
        y_pred_meta = np.append(y_pred_meta, y_pred_task)
        y_true_meta = np.append(y_true_meta, y_true_task)  

        if plot:          
          plt.plot(y_pred_task, y_true_task, 'b.')       
          y_vector = np.append(y_true_task, y_pred_task)
          min_list.append(np.amin(y_vector))
          max_list.append(np.amax(y_vector))
          
      if self.weighted_metric_of_each_endpoint:
        # It calculates the weighted average of the metric values for each single task.
        total_valid_obs_for_metatask = 0
        sum_metric_values = 0
        for task_name, num_obs_for_task in single_task_name_to_num_obs.items():
          single_task_value = single_task_name_to_metric_value[task_name]
          if np.isnan(single_task_value):
            continue
          total_valid_obs_for_metatask += num_obs_for_task
          sum_metric_values += num_obs_for_task * single_task_value
        if total_valid_obs_for_metatask > 0:
          metric_value = sum_metric_values/total_valid_obs_for_metatask  
        else:
          metric_value = np.nan
        
      else:
        # Calculates the metric value for the whole metatask which is represented as a concatenated vector.
        calculate_metric = True
        if len(y_true_meta) > 0:        
          if self.metric.__name__ in {'concordance_index', 'r2_score'}:
            # See whether the y_true vector has more than 0 valid pairs. If so, calculate. 
            unequal_count = sum(y_true_meta[0] != y_true_meta)
            if unequal_count <= 0:
              #pdb.set_trace()
              calculate_metric = False
              assert aggregated_num_obs is not None
              assert excluded_metatasks_dict is not None
              excluded_metatasks_dict[meta_task_name] = aggregated_num_obs[meta_task_name]              
        else:
          calculate_metric = False            
          
        if calculate_metric:
          metric_value = self.metric(y_true_meta, y_pred_meta)
        else:
          metric_value = np.nan
          
      computed_metrics.append(metric_value)
      if task_name_to_metric_value is not None:
        assert meta_task_name not in task_name_to_metric_value
        task_name_to_metric_value[meta_task_name] = metric_value
      
      if plot and len(y_true_meta) > 0:
        # Plotting for the whole metatask.
        min_value = np.amin(np.array(min_list)) 
        max_value = np.amax(np.array(max_list))         
        plt.plot([min_value-1, max_value + 1], [min_value-1, max_value + 1], 'k')      
        plt.xlabel("Predicted value")
        plt.ylabel("True value") 
        if is_training_set:
          meta_task_name = meta_task_name + "_trainset"
        else:
          np.save("plots/y_pred_" + meta_task_name, y_pred_meta)
          np.save("plots/y_true_" + meta_task_name, y_true_meta)
        if model_name is not None:
          meta_task_name = model_name + "_" + meta_task_name
        plot_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        image_name = "plots/" + meta_task_name + plot_time + ".png"
        plt.savefig(image_name)
        plt.close()
 
 
  def compute_metric(self,
                     y_true,
                     y_pred,
                     w=None,
                     n_classes=2,
                     filter_nans=False,
                     per_task_metrics=False,
                     plot=False,
                     all_metrics=None,
                     is_training_set=False,
                     no_concordance_index=True,
                     tasks=None,
                     model_name=None):
    """Compute a performance metric for each task.

    Parameters
    ----------
    y_true: np.ndarray
      An np.ndarray containing true values for each task.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task.
    w: np.ndarray, optional
      An np.ndarray containing weights for each datapoint.
    n_classes: int, optional
      Number of classes in data for classification tasks.
    filter_nans: bool, optional
      Remove NaN values in computed metrics
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.

    Returns
    -------
    A numpy nd.array containing metric values for each task.
    """
    
    if len(y_true.shape) > 1:
      n_samples, n_tasks = y_true.shape[0], y_true.shape[1]
    else:
      n_samples, n_tasks = y_true.shape[0], 1
    if tasks is not None:
      assert len(tasks) == n_tasks
    if self.mode == "classification":
      y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    else:
      y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    y_true = np.reshape(y_true, (n_samples, n_tasks))
    if w is None or len(w) == 0:
      w = np.ones_like(y_true)
    assert y_true.shape[0] == y_pred.shape[0] == w.shape[0]
    computed_metrics = []
    excluded_single_tasks_dict = {}
    task_name_to_metric_value = {}
    time_start = time.time()
    
    # WARNING: the following block works fine, but if you change a similar block in driver.py, it might cause
    # problems. Do not do that.
    excluded_metatasks_dict = {}
    metatask_to_task = {}
    taskind_to_metatask = {}
    meta_task_list = []
    aggregated_task_names = copy.deepcopy(tasks)
    if len(self.aggregate_list) > 0:
      assert tasks is not None      
      for meta_task_name in self.aggregate_list:        
        for i, task_name in enumerate(tasks):
          if not re.search(meta_task_name, task_name, re.I):
            continue
          # Only construct the corresponding entry when there are truly such hits.
          if meta_task_name not in metatask_to_task:
            metatask_to_task[meta_task_name] = []
            aggregated_task_names.append(meta_task_name)
          if i not in taskind_to_metatask:
            taskind_to_metatask[i] = meta_task_name
          if meta_task_name not in meta_task_list:
            meta_task_list.append(meta_task_name)
          aggregated_task_names.remove(task_name)
          pair = (i, task_name)
          metatask_to_task[meta_task_name].append(pair)            

    n_aggregated_tasks = len(aggregated_task_names)
    do_aggregation = len(taskind_to_metatask) > 0       

    if self.arithmetic_mean:
      # assert self.weighted_metric_of_each_endpoint
      total_datapoints = 0
      num_observations = []
      aggregated_num_obs = {meta_task_name: 0 for meta_task_name in metatask_to_task}
      for task in range(n_tasks):
        w_task = w[:, task]        
        this_datapoints = sum(w_task != 0)
        num_observations.append(this_datapoints)
        total_datapoints += this_datapoints
        if not do_aggregation:
          continue
        if task in taskind_to_metatask:
          meta_task_name = taskind_to_metatask[task]
          aggregated_num_obs[meta_task_name] += this_datapoints        
    
    self.get_metric_values_for_tasks(excluded_single_tasks_dict, computed_metrics, tasks, y_true, 
      y_pred, w, plot, taskind_to_metatask=taskind_to_metatask, num_observations=num_observations, 
      all_metrics=all_metrics, is_training_set=is_training_set, 
      no_concordance_index=no_concordance_index, model_name=model_name, do_aggregation=do_aggregation,
      task_name_to_metric_value=task_name_to_metric_value) 
    
    # Calculates the metric values of the tasks. Hopefully it is correct.
    if do_aggregation:
      self.plot_and_get_metric_values_for_metatasks(meta_task_list, metatask_to_task, computed_metrics,
        y_true, y_pred, w, plot=plot, all_metrics=all_metrics, is_training_set=is_training_set, 
        no_concordance_index=no_concordance_index, model_name=model_name,
        excluded_metatasks_dict=excluded_metatasks_dict, aggregated_num_obs=aggregated_num_obs,
        task_name_to_metric_value=task_name_to_metric_value, num_observations=num_observations) 
        
    # Check to make sure the computed_metrics contains the elements in the order of aggregated_task_names.
    for i, task_name in enumerate(aggregated_task_names):      
      metric_val = task_name_to_metric_value[task_name]
      np.testing.assert_almost_equal(metric_val, computed_metrics[i], decimal=5)     
    
    weighted_metrics = []
    # This block populates the weighted_metrics.
    if self.arithmetic_mean:
      # TODO: we should extract this block as a function.
    #def fill_weighted_metrics_list(self, weighted_metrics, total_datapoints, num_observations):
      excluded_datapoints = 0
      for task_ind in excluded_single_tasks_dict:
        excluded_datapoints += excluded_single_tasks_dict[task_ind]
      for meta_task_name in excluded_metatasks_dict:
        excluded_datapoints += excluded_metatasks_dict[meta_task_name]
      
      total_datapoints -= excluded_datapoints
      sum_coefficient = 0
      included_n_tasks = sum(np.invert(np.isnan(computed_metrics)))
      for task in range(n_tasks):
        # This block only processes the tasks that are not aggregated.
        if task in taskind_to_metatask:
          continue
        if task in excluded_single_tasks_dict:
          weighted_metrics.append(np.nan)
          continue
        task_coefficient = included_n_tasks * num_observations[task] / total_datapoints
        sum_coefficient += task_coefficient        
        weighted_metrics.append(task_coefficient * computed_metrics[task])
        
      # Now deal with aggregated tasks. 
      for meta_task_name in meta_task_list:
        if meta_task_name in excluded_metatasks_dict:
          weighted_metrics.append(np.nan)
          continue
        num_obs = aggregated_num_obs[meta_task_name]
        task_coefficient = included_n_tasks * num_obs / total_datapoints
        sum_coefficient += task_coefficient
        weighted_metrics.append(task_coefficient * task_name_to_metric_value[meta_task_name])        
      
      np.testing.assert_almost_equal(sum_coefficient, included_n_tasks, decimal=6)     

    time_end = time.time()             
        
    if not self.arithmetic_mean:
      weighted_metrics = copy.deepcopy(computed_metrics)
      assert not do_aggregation
      # if do_aggregation:
      #   warnings.warn("Aggregating in metric %s yet not using arithmetic mean. " % (self.metric.__name__))
      #   # TODO: calculating the simple average of the aggregated tasks

    if not do_aggregation:
      log("computed_metrics %s: %s" % (self.metric.__name__, str(computed_metrics)), self.verbose)
      log("corresponding to tasks: %s" % (str(tasks)), self.verbose)
    else:
      log("computed_metrics %s: %s" % (self.metric.__name__, str(computed_metrics)), self.verbose)
      log("corresponding to tasks: %s" % (str(aggregated_task_names)), self.verbose)
    print("time spent on evaluation: ", time_end - time_start)

    if n_tasks == 1:
      computed_metrics = computed_metrics[0]
    if not self.is_multitask:
      return computed_metrics
  
    if filter_nans:
      computed_metrics = np.array(computed_metrics)
      computed_metrics = computed_metrics[~np.isnan(computed_metrics)]
    if self.compute_energy_metric:
      # TODO(rbharath, joegomes): What is this magic number?
      force_error = self.task_averager(computed_metrics[1:]) * 4961.47596096
      print("Force error (metric: np.mean(%s)): %f kJ/mol/A" % (self.name,
                                                                force_error))
      return computed_metrics[0]
    if not per_task_metrics:
      return self.task_averager(weighted_metrics)
    if len(self.aggregate_list) > 0:
      assert do_aggregation
      return self.task_averager(weighted_metrics), computed_metrics
    
    return self.task_averager(weighted_metrics), computed_metrics
  

  def compute_singletask_metric(self, y_true, y_pred, w, plot=False, all_metrics=None,
    is_training_set=False, no_concordance_index=True, task_name=None, model_name=None):
    """Compute a metric value.

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.

    Returns:
      Float metric value.

    Raises:
      NotImplementedError: If metric_str is not in METRICS.
    """
    original_y_true = y_true
    original_y_pred = y_pred
    y_true = np.array(np.squeeze(y_true[w != 0]))
    y_pred = np.array(np.squeeze(y_pred[w != 0]))

    if len(y_true.shape) == 0:
      n_samples = 1
    else:
      n_samples = y_true.shape[0]
    # If there are no nonzero examples, metric is ill-defined.
    if not y_true.size:
      return np.nan

    y_true = np.reshape(y_true, (n_samples,))
    if self.mode == "classification":
      n_classes = y_pred.shape[-1]
      # TODO(rbharath): This has been a major source of bugs. Is there a more
      # robust characterization of which metrics require class-probs and which
      # don't?
      if "roc_auc_score" in self.name or "prc_auc_score" in self.name:
        y_true = to_one_hot(y_true).astype(int)
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
      else:
        y_true = y_true.astype(int)
        # Reshape to handle 1-d edge cases
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
        y_pred = from_one_hot(y_pred)
    else:
      y_pred = np.reshape(y_pred, (n_samples,))

    if self.threshold is not None:
      y_pred = np.greater(y_pred, self.threshold) * 1
      y_true = np.greater(y_true, self.threshold) * 1
      n_classes = 2
      y_pred = to_one_hot(y_pred).astype(int)
      y_true = to_one_hot(y_true).astype(int)      
      
    try:
      metric_value = self.metric(y_true, y_pred)
      if plot:
        metric_value_dict = {self.metric.__name__: metric_value}
        residuals = y_pred - y_true
             
        for some_metric in all_metrics:            
          if some_metric.metric.__name__ == self.metric.__name__:
            continue
          calculate_metric = True
          if is_training_set and no_concordance_index:
            if some_metric.metric.__name__ == "concordance_index":
              continue
          elif some_metric.metric.__name__ in {"concordance_index", 'r2_score'}:
            calculate_metric = (sum(y_true[0] != y_true) >= 1)
          if calculate_metric:                            
            some_metric_value = some_metric.compute_singletask_metric(original_y_true, 
              original_y_pred, w, plot=False)
          else:
            some_metric_value = np.nan
          metric_value_dict[some_metric.metric.__name__] = some_metric_value

        plt.plot(y_pred, y_true, 'b.')
        y_vector = np.append(y_true, y_pred)
        min_value = np.amin(y_vector)
        max_value = np.amax(y_vector)        
        plt.plot([min_value-1, max_value + 1], [min_value-1, max_value + 1], 'k')
        
        plt.xlabel("Predicted value")
        plt.ylabel("True value")        
        i = 0
        for some_metric in all_metrics:
          if some_metric.metric.__name__ in metric_value_dict:            
            plt.text(min_value - 1, max_value + 0.4 + 0.7*i, '%s=%f, ' % (some_metric.metric.__name__,
              metric_value_dict[some_metric.metric.__name__]))
            i += 1
        plot_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        if is_training_set:
          task_name = task_name + "_trainset"
        else:
          np.save("plots/y_pred", y_pred)
          np.save("plots/y_true", y_true)
          np.save("plots/residuals", residuals)
        if model_name is not None:
          task_name = model_name + "_" + task_name
        image_name = "plots/" + task_name + plot_time + ".png"
        plt.savefig(image_name)
        plt.close()

        # plt.plot(y_pred, residuals, 'r.')
        # plt.plot([min_value - 1, max_value + 1], [0, 0], 'k')        
        # plt.xlabel("Predicted value")
        # plt.ylabel("Residuals")  
        # task_name = "res_" + task_name
        # image_name2 = "plots/" + task_name + plot_time + ".png"
        # plt.savefig(image_name2)
        # plt.close()

    except (AssertionError, ValueError) as e:
      warnings.warn("Error calculating metric %s: %s" % (self.name, e))
      metric_value = np.nan
    return metric_value
