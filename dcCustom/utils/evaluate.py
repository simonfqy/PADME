"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import csv
import numpy as np
from deepchem.utils.save import log
from dcCustom.trans import undo_transforms
from dcCustom.metrics import from_one_hot

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"


def relative_difference(x, y):
  """Compute the relative difference between x and y"""
  return np.abs(x - y) / np.abs(max(x, y))


def threshold_predictions(y, threshold):
  y_out = np.zeros_like(y)
  for ind, pred in enumerate(y):
    y_out[ind] = 1 if pred > threshold else 0
  return y_out


# TODO(rbharath): This is now simple enough that we should probably get rid of
# Evaluator object to avoid clutter.
class Evaluator(object):
  """Class that evaluates a model on a given dataset."""

  def __init__(self, model, dataset, transformers, verbose=False, is_training_set=False,
    tasks=None, model_name=None):
    self.model = model
    self.dataset = dataset
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    self.task_names = dataset.get_task_names()
    self.verbose = verbose
    self.is_training_set = is_training_set
    self.tasks = tasks
    self.model_name = model_name

  def output_statistics(self, scores, stats_out):
    """
    Write computed stats to file.
    """
    with open(stats_out, "w") as statsfile:
      statsfile.write(str(scores) + "\n")

  def output_predictions(self, y_preds, csv_out):
    """
    Writes predictions to file.

    Args:
      y_preds: np.ndarray
      csvfile: Open file object.
    """
    mol_ids = self.dataset.ids
    n_tasks = len(self.task_names)
    y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
    assert len(y_preds) == len(mol_ids)
    with open(csv_out, "w") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["Compound"] + self.dataset.get_task_names())
      for mol_id, y_pred in zip(mol_ids, y_preds):
        csvwriter.writerow([mol_id] + list(y_pred))

  def compute_model_performance(self,
                                metrics,
                                csv_out=None,
                                stats_out=None,
                                per_task_metrics=False,
                                no_concordance_index=False,
                                plot=False,
                                no_r2=False):
    """
    Computes statistics of model on test data and saves results to csv.

    Parameters
    ----------
    metrics: list
      List of dc.metrics.Metric objects
    csv_out: str, optional
      Filename to write CSV of model predictions.
    stats_out: str, optional
      Filename to write computed statistics.
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    """
    y = self.dataset.y
    y = undo_transforms(y, self.output_transformers)
    w = self.dataset.w

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    y_pred = self.model.predict(self.dataset, self.output_transformers)
    if mode == "classification":
      y_pred_print = np.argmax(y_pred, -1)
    else:      
      y_pred_print = y_pred
    multitask_scores = {}
    all_task_scores = {}

    if csv_out is not None:
      log("Saving predictions to %s" % csv_out, self.verbose)
      self.output_predictions(y_pred_print, csv_out)

    plot_finished = False
    # Compute multitask metrics
    for i, metric in enumerate(metrics):
      mtc_name = metric.metric.__name__
      if no_r2 and (mtc_name == 'r2_score' or mtc_name == 'pearson_r2_score'):
        continue
      if per_task_metrics:
        if self.is_training_set:
          if no_concordance_index and metric.metric.__name__ =="concordance_index":
            multitask_scores[metric.name] = None
            all_task_scores[metric.name] = None
            continue
          if plot and not plot_finished:
            # If this dataset is the training data set, don't calculate CI if no_concordance_index.
            multitask_scores[metric.name], computed_metrics = metric.compute_metric(
                y, y_pred, w, per_task_metrics=True, plot=True, all_metrics=metrics,
                is_training_set=self.is_training_set, no_concordance_index=no_concordance_index,
                tasks=self.tasks, model_name=self.model_name)
            all_task_scores[metric.name] = computed_metrics
            plot_finished = True
          else:
            # No longer need to plot. Could be wasting time calculating metrics again, but they
            # are super fast so it is no big deal.
            multitask_scores[metric.name], computed_metrics = metric.compute_metric(
                y, y_pred, w, per_task_metrics=True, plot=False, 
                is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
            all_task_scores[metric.name] = computed_metrics

        # Now deal with validation or test sets.
        elif plot and (i == len(metrics)-1 or metric.metric.__name__ =="concordance_index") and (
          not plot_finished):                  
          multitask_scores[metric.name], computed_metrics = metric.compute_metric(
              y, y_pred, w, per_task_metrics=True, plot=True, all_metrics=metrics,
              is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
          all_task_scores[metric.name] = computed_metrics
          plot_finished = True
        else: # Otherwise don't need to plot.
          multitask_scores[metric.name], computed_metrics = metric.compute_metric(
              y, y_pred, w, per_task_metrics=True, plot=False, is_training_set=self.is_training_set, 
              tasks=self.tasks, model_name=self.model_name)
          all_task_scores[metric.name] = computed_metrics

      else:        
        if self.is_training_set:
          if no_concordance_index and metric.metric.__name__ =="concordance_index":
            multitask_scores[metric.name] = None            
            continue
          if plot and not plot_finished:
            multitask_scores[metric.name] = metric.compute_metric(
                y, y_pred, w, per_task_metrics=False, plot=True, all_metrics=metrics,
                is_training_set=self.is_training_set, no_concordance_index=no_concordance_index, 
                tasks=self.tasks, model_name=self.model_name)            
            plot_finished = True
          else:            
            multitask_scores[metric.name] = metric.compute_metric(
                y, y_pred, w, per_task_metrics=False, plot=False,
                is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
            
        elif plot and (i == len(metrics)-1 or metric.metric.__name__ =="concordance_index") and (
          not plot_finished):                  
          multitask_scores[metric.name] = metric.compute_metric(
              y, y_pred, w, per_task_metrics=False, plot=True, all_metrics=metrics,
              is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
          plot_finished = True
        else:
          multitask_scores[metric.name] = metric.compute_metric(
              y, y_pred, w, per_task_metrics=False, plot=False,
              is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)

    if stats_out is not None:
      log("Saving stats to %s" % stats_out, self.verbose)
      self.output_statistics(multitask_scores, stats_out)

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores


class GeneratorEvaluator(object):
  """
  Partner class to Evaluator.
  Instead of operating over datasets this class operates over Generator.
  Evaluate a Metric over a model and Generator.
  """

  def __init__(self,
               model,
               generator,
               transformers,
               labels,
               dataset=None,
               outputs=None,
               n_tasks=1,
               n_classes=2,
               weights=list(),
               is_training_set=False,
               tasks=None,
               model_name=None):
    """
    Parameters
    ----------
    model: Model
      Model to evaluate
    generator: Generator
      Generator which yields {layer: numpyArray} to feed into model
    transformers:
      Tranformers to "undo" when applied to the models outputs
    labels: list of Layer
      layers which are keys in the generator to compare to outputs
    outputs: list of Layer
      if None will use the outputs of the model
    weights: np.array
      Must be of the shape (n_samples, n_tasks)
      if weights[sample][task] is 0 that sample will not be used
      for computing the task metric
    """
    self.model = model
    self.generator = generator
    self.n_tasks = n_tasks
    self.n_classes = n_classes
    # I added the dataset here to implement output_predictions() function easier.
    self.dataset = dataset
    self.output_transformers = [
        transformer for transformer in transformers if transformer.transform_y
    ]
    if outputs is None:
      self.output_keys = model.outputs
    else:
      self.output_keys = outputs
    self.label_keys = labels
    self.weights = weights
    self.is_training_set = is_training_set
    self.tasks = tasks
    self.model_name = model_name
    if len(self.label_keys) != len(self.output_keys):
      raise ValueError("Must have same number of labels and outputs")
    if len(self.label_keys) != 1:
      raise ValueError("GeneratorEvaluator currently only supports one label.")

  # TODO: the following function needs revision to work properly.
  def output_predictions(self, y_preds, csv_out):
    """
    Writes predictions to file.

    Args:
      y_preds: np.ndarray
      csvfile: Open file object.
    """
    mol_ids = self.dataset.ids
    y_preds = np.reshape(y_preds, (len(y_preds), self.n_tasks))
    assert len(y_preds) == len(mol_ids)
    with open(csv_out, "w") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["Compound"] + self.dataset.get_task_names())
      for mol_id, y_pred in zip(mol_ids, y_preds):
        csvwriter.writerow([mol_id] + list(y_pred))

  def compute_model_performance(self, metrics, csv_out=None, stats_out=None, 
                                per_task_metrics=False, no_r2=False,
                                no_concordance_index=False, plot=False):
    """
    Computes statistics of model on test data and saves results to csv.

    Parameters
    ----------
    metrics: list
      List of dc.metrics.Metric objects
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    """
    self.model.build()
    y = []
    w = []

    def generator_closure():
      for feed_dict in self.generator:
        y.append(feed_dict[self.label_keys[0]])
        if len(self.weights) > 0:
          w.append(feed_dict[self.weights[0]])
        yield feed_dict

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    if mode == "classification":
      y_pred = self.model.predict_proba_on_generator(generator_closure())
      y = np.transpose(np.array(y), axes=[0, 2, 1, 3])
      y = np.reshape(y, newshape=(-1, self.n_tasks, self.n_classes))
      y = from_one_hot(y, axis=-1)
    else:
      y_pred = self.model.predict_proba_on_generator(generator_closure())
      y = np.transpose(np.array(y), axes=[0, 2, 1, 3])
      y = np.reshape(y, newshape=(-1, self.n_tasks))
      y_pred = np.reshape(y_pred, newshape=(-1, self.n_tasks))
    y_pred = self.model.predict_on_generator(generator_closure())
    y = np.concatenate(y, axis=0)
    multitask_scores = {}
    all_task_scores = {}

    y = undo_transforms(y, self.output_transformers)
    y_pred = undo_transforms(y_pred, self.output_transformers)
    if len(w) != 0:
      w = np.array(w)
      w = np.reshape(w, newshape=y.shape)

    if csv_out is not None:
      log("Saving predictions to %s" % csv_out, self.verbose)
      self.output_predictions(y_pred, csv_out)

    plot_finished = False
    # Compute multitask metrics
    for i, metric in enumerate(metrics):
      mtc_name = metric.metric.__name__
      if no_r2 and (mtc_name == 'r2_score' or mtc_name == 'pearson_r2_score'):
        continue
      if per_task_metrics:
        if self.is_training_set:
          if no_concordance_index and metric.metric.__name__ =="concordance_index":
            multitask_scores[metric.name] = None
            all_task_scores[metric.name] = None
            continue
          if plot and not plot_finished:
            multitask_scores[metric.name], computed_metrics = metric.compute_metric(
              y, y_pred, w, per_task_metrics=True, n_classes=self.n_classes, plot=True,
              all_metrics=metrics, is_training_set=self.is_training_set,
              no_concordance_index=no_concordance_index, tasks=self.tasks, 
              model_name=self.model_name)
            all_task_scores[metric.name] = computed_metrics
            plot_finished = True
          else:
            multitask_scores[metric.name], computed_metrics = metric.compute_metric(
              y, y_pred, w, per_task_metrics=True, n_classes=self.n_classes, plot=False,
              is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
            all_task_scores[metric.name] = computed_metrics

        elif plot and (i == len(metrics)-1 or metric.metric.__name__ =="concordance_index") and (
          not plot_finished):                  
          multitask_scores[metric.name], computed_metrics = metric.compute_metric(
              y, y_pred, w, per_task_metrics=True, n_classes=self.n_classes, plot=True, 
              all_metrics=metrics, is_training_set=self.is_training_set, tasks=self.tasks, 
              model_name=self.model_name)
          all_task_scores[metric.name] = computed_metrics
          plot_finished = True

        else: #Otherwise don't need to plot.
          multitask_scores[metric.name], computed_metrics = metric.compute_metric(
            y, y_pred, w, per_task_metrics=True, n_classes=self.n_classes, plot=False, 
            is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)
          all_task_scores[metric.name] = computed_metrics

      else:
        if self.is_training_set:
          if no_concordance_index and metric.metric.__name__ =="concordance_index":
            multitask_scores[metric.name] = None
            continue
          if plot and not plot_finished:
            multitask_scores[metric.name] = metric.compute_metric(
              y, y_pred, w, per_task_metrics=False, n_classes=self.n_classes, plot=True,
              all_metrics=metrics, is_training_set=self.is_training_set,
              no_concordance_index=no_concordance_index, tasks=self.tasks, 
              model_name=self.model_name)
            plot_finished = True
          else:
            multitask_scores[metric.name] = metric.compute_metric(
              y, y_pred, w, per_task_metrics=False, n_classes=self.n_classes, plot=False,
              is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)

        elif plot and (i == len(metrics)-1 or metric.metric.__name__ =="concordance_index") and (
          not plot_finished):                  
          multitask_scores[metric.name] = metric.compute_metric(
              y, y_pred, w, per_task_metrics=False, n_classes=self.n_classes, plot=True, 
              all_metrics=metrics, is_training_set=self.is_training_set, tasks=self.tasks, 
              model_name=self.model_name)
          plot_finished = True

        else: #Otherwise don't need to plot.
          multitask_scores[metric.name]= metric.compute_metric(
            y, y_pred, w, per_task_metrics=False, n_classes=self.n_classes, plot=False,
            is_training_set=self.is_training_set, tasks=self.tasks, model_name=self.model_name)          

    if not per_task_metrics:
      return multitask_scores
    else:
      return multitask_scores, all_task_scores
