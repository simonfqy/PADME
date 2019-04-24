from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os
import time
import sys
import pdb
import csv
import re
import copy
import pickle
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification
from dcCustom.molnet.check_availability import CheckFeaturizer, CheckSplit

FLAGS = None

def get_aggregate_list(aggregate_suffix_file, aggregate):
  assert isinstance(aggregate, list)
  assert len(aggregate) > 0
  prefix = aggregate[0] + '_'
  df_subgroup = pd.read_csv(aggregate_suffix_file, header = 0, index_col=0)
  subgroup_list = list(set(df_subgroup.iloc[:]['Subgroup']))
  new_aggregate_list = [prefix + str(subgroup) for subgroup in subgroup_list]
  return new_aggregate_list


def load_prot_dict(prot_desc_dict, prot_seq_dict, prot_desc_path, 
  sequence_field, phospho_field):
  if re.search('davis', prot_desc_path, re.I):
    source = 'davis'
  elif re.search('metz', prot_desc_path, re.I):
    source = 'metz'
  elif re.search('kiba', prot_desc_path, re.I):
    source = 'kiba'
  elif re.search('toxcast', prot_desc_path, re.I):
    source = 'toxcast'

  df = pd.read_csv(prot_desc_path, index_col=0)
  #protList = list(df.index)  
  for row in df.itertuples():
    descriptor = row[2:]
    descriptor = np.array(descriptor)
    descriptor = np.reshape(descriptor, (1, len(descriptor)))
    pair = (source, row[0])
    assert pair not in prot_desc_dict
    prot_desc_dict[pair] = descriptor
    sequence = row[sequence_field]
    phosphorylated = row[phospho_field]
    assert pair not in prot_seq_dict
    prot_seq_dict[pair] = (phosphorylated, sequence)   
    
    
def write_file_for_a_cv_iteration(writer, train_score, valid_score, model_name, dataset, h, tasks,
  aggregated_tasks, time_finish=None, time_start=None):
  
  train_score_dict = train_score[model_name]
  valid_score_dict = valid_score[model_name]
  
  if len(tasks) > 1:
    train_score_dict = train_score[model_name]['averaged']
    valid_score_dict = valid_score[model_name]['averaged']          

  for i in train_score_dict:
    # i here is the metric name, like 'rms_score'.        
    this_train_score = train_score_dict[i]
    this_valid_score = valid_score_dict[i] 
    output_line = [
          dataset,
          model_name, i, 'train',
          this_train_score, 'valid', this_valid_score,
          'fold_num', h
    ]   
    if time_finish is not None and time_start is not None:
      output_line.extend(['time_for_running', time_finish - time_start])
    writer.writerow(output_line)
    
    if len(aggregated_tasks) > 1:
      train_score_tasks = train_score[model_name]['per_task_score'][i]
      valid_score_tasks = valid_score[model_name]['per_task_score'][i]
      
      for index, task in enumerate(aggregated_tasks):
        train_sc_tk = None if train_score_tasks is None else train_score_tasks[index]
        dataset_nm = dataset + '_' + task
        output_line = [
                dataset_nm,
                model_name, i, 'train',
                train_sc_tk, 'valid', valid_score_tasks[index]
        ]              
        writer.writerow(output_line)


def write_intermediate_file(out_path, intermediate_file, train_scores_list, 
  valid_scores_list, train_score, valid_score, tasks, dataset, h, aggregated_tasks):
  
  train_scores_list.append(train_score)
  valid_scores_list.append(valid_score)
  with open(os.path.join(out_path, intermediate_file), 'a') as f:
    writer = csv.writer(f)
    model_name = list(train_scores_list[0].keys())[0]
    
    write_file_for_a_cv_iteration(writer, train_score, valid_score, model_name, dataset, h, tasks,
      aggregated_tasks, time_finish=None, time_start=None)


def write_avg_to_interm_file(out_path, intermediate_file, fold_num, train_scores_list, 
  valid_scores_list, tasks, dataset, h='CV_average'):
  
  model_name = list(train_scores_list[0].keys())[0]
  num_iteration = len(valid_scores_list)
  if num_iteration != fold_num:
    return
  train_metric_name_to_value_sum = dict() 
  valid_metric_name_to_value_sum = dict()     

  for train_score, valid_score in zip(train_scores_list, valid_scores_list):
    train_score_dict = train_score[model_name]
    valid_score_dict = valid_score[model_name]
    
    if len(tasks) > 1:
      train_score_dict = train_score[model_name]['averaged']
      valid_score_dict = valid_score[model_name]['averaged']          

    for i in train_score_dict:
      # i here is the metric name, like 'rms_score'.        
      this_train_score = train_score_dict[i]
      this_valid_score = valid_score_dict[i] 
      if i not in train_metric_name_to_value_sum:
        train_metric_name_to_value_sum[i] = 0
        valid_metric_name_to_value_sum[i] = 0
      train_metric_name_to_value_sum[i] += this_train_score
      valid_metric_name_to_value_sum[i] += this_valid_score      

  with open(os.path.join(out_path, intermediate_file), 'a') as f:
    writer = csv.writer(f)
    for i in train_metric_name_to_value_sum:
      train_score_avg = train_metric_name_to_value_sum[i] / num_iteration
      valid_score_avg = valid_metric_name_to_value_sum[i] / num_iteration
      output_line = [
              dataset,
              model_name, i, 'train',
              train_score_avg, 'valid', valid_score_avg,
              'fold_num', h
      ]
      writer.writerow(output_line)
      
      
def write_results_file(out_path, results_file, train_scores_list, valid_scores_list, fold_num,
  dataset, tasks, aggregated_tasks, time_finish_fitting, time_start_fitting, cross_validation=True,
  test=False, test_scores_list=None, early_stopping=False, opt_epoch=None):
  
  with open(os.path.join(out_path, results_file), 'a') as f:
    writer = csv.writer(f)
    model_name = list(train_scores_list[0].keys())[0]
     
    if cross_validation:
      for h in range(fold_num):
        train_score = train_scores_list[h]
        valid_score = valid_scores_list[h]
        
        write_file_for_a_cv_iteration(writer, train_score, valid_score, model_name, dataset, h, 
          tasks, aggregated_tasks, time_finish=time_finish_fitting, time_start=time_start_fitting)          
          
    else:
      train_score = train_scores_list[0]
      valid_score = valid_scores_list[0]
      if test and test_scores_list is not None:
        test_score = test_scores_list[0]

      train_score_dict = train_score[model_name]
      valid_score_dict = valid_score[model_name]
      if test:
        test_score_dict = test_score[model_name]
      if len(tasks) > 1:
        train_score_dict = train_score[model_name]['averaged']
        valid_score_dict = valid_score[model_name]['averaged']
        if test:
          test_score_dict = test_score[model_name]['averaged']

      for i in train_score_dict:
        # i here is the metric name, like 'rms_score'.        
        this_train_score = train_score_dict[i]
        this_valid_score = valid_score_dict[i]
        if test:
          this_test_score = test_score_dict[i]
              
        output_line = [
                  dataset,
                  model_name, i, 'train',
                  this_train_score, 'valid', this_valid_score
        ]
        if test:
          output_line.extend(['test', this_test_score])
        if early_stopping and opt_epoch is not None:
          output_line.extend(['optimal epoch', opt_epoch[0]])
          output_line.extend(['best validation score', opt_epoch[1]])
        writer.writerow(output_line)
        
        if len(aggregated_tasks) > 1:
          train_score_tasks = train_score[model_name]['per_task_score'][i]
          valid_score_tasks = valid_score[model_name]['per_task_score'][i]
          if test:
            test_score_tasks = test_score[model_name]['per_task_score'][i]
          for index, task in enumerate(aggregated_tasks):
            train_sc_tk = None if train_score_tasks is None else train_score_tasks[index]
            dataset_nm = dataset + '_' + task
            output_line = [
                    dataset_nm,
                    model_name, i, 'train',
                    train_sc_tk, 'valid', valid_score_tasks[index]
            ]
            if test:
              output_line.extend(['test', test_score_tasks[index]])            
            writer.writerow(output_line)
          

def run_analysis(_):
  dataset = FLAGS.dataset 
  model = FLAGS.model
  thresholding = FLAGS.thresholding
  split= FLAGS.split
  threshold = FLAGS.threshold
  no_concord = FLAGS.no_concord
  no_r2 = FLAGS.no_r2
  #direction = FLAGS.direction
  out_path = FLAGS.out_path
  fold_num = FLAGS.fold_num
  hyper_parameters=FLAGS.hyper_parameters
  hyper_param_search = FLAGS.hyper_param_search
  verbose_search = FLAGS.verbose_search
  arithmetic_mean = FLAGS.arithmetic_mean
  max_iter = FLAGS.max_iter
  search_range = FLAGS.search_range
  isreload = FLAGS.reload
  cross_validation = FLAGS.cross_validation
  test = FLAGS.test
  predict_cold = FLAGS.predict_cold # Determines whether cold-start
  #drugs and targets are tested or validated.
  cold_drug = FLAGS.cold_drug
  cold_target = FLAGS.cold_target
  cold_drug_cluster = FLAGS.cold_drug_cluster
  split_warm = FLAGS.split_warm
  filter_threshold = FLAGS.filter_threshold
  early_stopping = FLAGS.early_stopping
  evaluate_freq = FLAGS.evaluate_freq # Number of training epochs before evaluating
  # for early stopping.
  patience = FLAGS.patience
  seed=FLAGS.seed
  log_file = FLAGS.log_file
  model_dir = FLAGS.model_dir
  prot_desc_path=FLAGS.prot_desc_path
  intermediate_file = FLAGS.intermediate_file
  plot = FLAGS.plot
  aggregate = FLAGS.aggregate
  aggregate_suffix_file = FLAGS.aggregate_suffix_file
  predict_only = FLAGS.predict_only
  restore_model = FLAGS.restore_model
  csv_out = FLAGS.csv_out
  tensorboard = FLAGS.tensorboard
  oversampled = FLAGS.oversampled
  input_protein = not FLAGS.no_input_protein
  weighted_metric_of_each_endpoint = FLAGS.weighted_metric_of_each_endpoint
  remove_val_set_entries = FLAGS.remove_val_set_entries
  
  if aggregate_suffix_file is not None and len(aggregate) > 0:
    aggregate = get_aggregate_list(aggregate_suffix_file, aggregate)
  assert len(set(aggregate)) == len(aggregate)

  if predict_only:
    hyper_param_search = False
    cross_validation = False
    plot = False
    early_stopping = False
    test = False 
    restore_model = True               
  else:  
    assert (predict_cold + cold_drug + cold_target + split_warm + cold_drug_cluster) <= 1

  assert model == model # Not a NAN
  searchObj = re.search('reg', model, re.I)
  mode = 'regression' if searchObj else 'classification'
  if mode == 'regression':
    if thresholding:
      mode = 'reg-threshold'
  direction = False

  if mode == 'regression':
    metrics = [dcCustom.metrics.Metric(dcCustom.metrics.rms_score, np.nanmean, 
      arithmetic_mean=arithmetic_mean, aggregate_list=aggregate, 
      weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint),
      dcCustom.metrics.Metric(dcCustom.metrics.concordance_index, np.nanmean, 
      arithmetic_mean=arithmetic_mean, aggregate_list=aggregate,
      weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint),
      dcCustom.metrics.Metric(dcCustom.metrics.r2_score, np.nanmean, 
      arithmetic_mean=arithmetic_mean, aggregate_list=aggregate,
      weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint)]
  elif mode == 'classification':
    direction = True
    metrics = [dcCustom.metrics.Metric(dcCustom.metrics.roc_auc_score, np.nanmean, 
      arithmetic_mean=arithmetic_mean, aggregate_list=aggregate,
      weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint),
      dcCustom.metrics.Metric(dcCustom.metrics.prc_auc_score, np.nanmean, 
      arithmetic_mean=arithmetic_mean, aggregate_list=aggregate,
      weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint)]
  elif mode == "reg-threshold":
    # TODO: this [0] is just a temporary solution. Need to implement per-task thresholds.
    # It is not a very trivial task.
    direction = True
    metrics = [dcCustom.metrics.Metric(dcCustom.metrics.roc_auc_score, np.nanmean, 
      threshold=threshold[0], mode="regression", arithmetic_mean=arithmetic_mean, 
      aggregate_list=aggregate, weighted_metric_of_each_endpoint=weighted_metric_of_each_endpoint)]

  # We assume that values above the threshold are "on" or 1, those below are "off"
  # or 0.  
  loading_functions = {
    'davis': dcCustom.molnet.load_davis,
    'metz': dcCustom.molnet.load_metz,
    'kiba': dcCustom.molnet.load_kiba,
    'toxcast': dcCustom.molnet.load_toxcast,
    'all_kinase': dcCustom.molnet.load_kinases,
    'tc_kinase':dcCustom.molnet.load_tc_kinases,
    'tc_full_kinase': dcCustom.molnet.load_tc_full_kinases,
    'nci60': dcCustom.molnet.load_nci60
  }  
  # if mode == 'regression' or mode == 'reg-threshold':
    # model = 'weave_regression'
    
  pair = (dataset, model)
  if pair in CheckFeaturizer:
    featurizer = CheckFeaturizer[pair][0]
    n_features = CheckFeaturizer[pair][1]

  if not split in [None] + CheckSplit[dataset]:
    return

  print('-------------------------------------')
  print('Running on dataset: %s' % dataset)
  print('-------------------------------------')

  prot_desc_dict = {}
  prot_seq_dict = {}
  if input_protein:
    for path in prot_desc_path:
      load_prot_dict(prot_desc_dict, prot_seq_dict, path, 1, 2)
  prot_desc_length = 8421
  
  if cross_validation:    
    tasks, all_dataset, transformers = loading_functions[dataset](featurizer=featurizer, 
                                                  cross_validation=cross_validation,
                                                  test=test, split=split, reload=isreload, 
                                                  K = fold_num, mode=mode, predict_cold=predict_cold,
                                                  cold_drug=cold_drug, cold_target=cold_target, 
                                                  cold_drug_cluster=cold_drug_cluster, split_warm=split_warm, 
                                                  prot_seq_dict=prot_seq_dict, 
                                                  filter_threshold=filter_threshold, oversampled=oversampled,
                                                  input_protein=input_protein, 
                                                  remove_val_set_entries=remove_val_set_entries)
  else:
    tasks, all_dataset, transformers = loading_functions[dataset](featurizer=featurizer, 
                                                  cross_validation=cross_validation,
                                                  test=test, split=split, reload=isreload, mode=mode,
                                                  predict_cold=predict_cold, cold_drug=cold_drug, 
                                                  cold_target=cold_target, cold_drug_cluster=cold_drug_cluster, 
                                                  split_warm=split_warm, filter_threshold=filter_threshold, 
                                                  prot_seq_dict=prot_seq_dict, oversampled=oversampled,
                                                  input_protein=input_protein,
                                                  remove_val_set_entries=remove_val_set_entries)
    
  # all_dataset will be a list of 5 elements (since we will use 5-fold cross validation),
  # each element is a tuple, in which the first entry is a training dataset, the second is
  # a validation dataset.
  # assert False
  time_start_fitting = time.time()
  train_scores_list = []
  valid_scores_list = []
  test_scores_list = []

  aggregated_tasks = copy.deepcopy(tasks)
  meta_task_list = []
  if aggregate is not None and len(aggregate) > 0:
    assert tasks is not None      
    for meta_task_name in aggregate:        
      for i, task_name in enumerate(tasks):
        if not re.search(meta_task_name, task_name, re.I):  
          continue                
        if meta_task_name not in meta_task_list:
          meta_task_list.append(meta_task_name)
          aggregated_tasks.append(meta_task_name)          
        aggregated_tasks.remove(task_name)

  matchObj = re.match('mpnn', model, re.I)
  model = 'mpnn' if matchObj else model
  
  if hyper_param_search: # We don't use cross validation in this case.
    if hyper_parameters is None:
      hyper_parameters = hps[model]
    train_dataset, valid_dataset, test_dataset = all_dataset
    search_mode = dcCustom.hyper.GaussianProcessHyperparamOpt(model)
    hyper_param_opt, _ = search_mode.hyperparam_search(
        hyper_parameters,
        train_dataset,
        valid_dataset,
        transformers,
        metrics,
        prot_desc_dict,
        prot_desc_length,
        tasks=tasks,
        direction=direction,
        n_features=n_features,
        n_tasks=len(tasks),
        max_iter=max_iter,
        search_range=search_range,
        early_stopping = early_stopping,
        evaluate_freq=evaluate_freq,
        patience=patience,
        model_dir=model_dir,
        log_file=log_file,
        tensorboard=tensorboard,
        mode=mode,
        no_concordance_index=no_concord,
        no_r2=no_r2,
        plot=plot,
        verbose_search=verbose_search,
        aggregated_tasks=aggregated_tasks,
        input_protein=input_protein)
    hyper_parameters = hyper_param_opt
  
  opt_epoch = -1
  test_dataset = None
  model_functions = {
    'regression': model_regression,
    'reg-threshold': model_regression,
    'classification': model_classification
  }
  assert mode in model_functions
  if mode == 'classification':
    direction=True
  
  if not cross_validation:
    train_dataset, valid_dataset, test_dataset = all_dataset
    train_score, valid_score, test_score, opt_epoch = model_functions[mode](
          train_dataset,
          valid_dataset,
          test_dataset,
          tasks,
          transformers,
          n_features,
          metrics,
          model,
          prot_desc_dict,
          prot_desc_length,
          hyper_parameters=hyper_parameters,
          test = test,         
          early_stopping = early_stopping,
          evaluate_freq = evaluate_freq, # Number of training epochs before evaluating
          # for early stopping.
          patience = patience,
          direction=direction,
          seed=seed,
          model_dir=model_dir,
          no_concordance_index=no_concord,
          no_r2=no_r2,
          plot=plot,
          aggregated_tasks=aggregated_tasks,
          tensorboard=tensorboard,
          predict_only=predict_only,
          restore_model=restore_model,
          prediction_file=csv_out,
          input_protein=input_protein)
    if predict_only:
      return
    train_scores_list.append(train_score)
    valid_scores_list.append(valid_score)
    test_scores_list.append(test_score)
  else:
    for h in range(fold_num):  
      train_score, valid_score, _, _ = model_functions[mode](
          all_dataset[h][0],
          all_dataset[h][1],
          None,
          tasks,
          transformers,
          n_features,
          metrics,
          model,
          prot_desc_dict,
          prot_desc_length,
          hyper_parameters=hyper_parameters,
          test = test,
          early_stopping = False,
          direction=direction,
          seed=seed,
          model_dir=model_dir,
          no_concordance_index=no_concord,
          tensorboard=tensorboard,
          no_r2=no_r2,
          plot=plot,
          aggregated_tasks=aggregated_tasks,
          restore_model=restore_model,
          input_protein=input_protein)
      # I made the decision to force disable early stopping for cross validation here,
      # not quite sure whether this is right.      
       
      # The section below is a workaround for the instability of the server. I don't like
      # it but guess there is no other choices.
      write_intermediate_file(out_path, intermediate_file, train_scores_list, 
        valid_scores_list, train_score, valid_score, tasks, dataset, h, aggregated_tasks)  

    write_avg_to_interm_file(out_path, intermediate_file, fold_num, train_scores_list, 
      valid_scores_list, tasks, dataset, h='CV_average')    
  
  time_finish_fitting = time.time()
  
  results_file = './results/results_' + model
  
  if mode == 'classification':
    results_file += '_cls'
  elif mode == 'reg-threshold':
    results_file += '_thrhd'
  if predict_cold:
    results_file += '_cold'
  if split_warm:
    results_file += '_warm'
  if cold_drug:
    results_file += '_cold_drug'
  elif cold_target:
    results_file += '_cold_target'
  if cold_drug_cluster:
    results_file += '_cold_drug_cluster'
  if cross_validation:
    results_file += '_cv'

  results_file += '.csv'

  write_results_file(out_path, results_file, train_scores_list, valid_scores_list, fold_num,
    dataset, tasks, aggregated_tasks, time_finish_fitting, time_start_fitting, 
    cross_validation=cross_validation, test=test, test_scores_list=test_scores_list, 
    early_stopping=early_stopping, opt_epoch=opt_epoch)
    
  if hyper_param_search:
    with open(os.path.join(out_path, dataset + model + '.pkl'), 'w') as f:
      pickle.dump(hyper_parameters, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset', type=str, default='davis', help='Dataset name.'
  )
  parser.add_argument(
      '--model', type=str, default='weave_regression', help='Name of the model.'
  )
  parser.add_argument(
      '--thresholding',
      default=False,
      help='If true, then it is a thresholding problem.',
      action='store_true'
  )
  parser.add_argument(
      '--no_concord',
      default=False,
      help='If true, then concordance index will not be computed for the training set.',
      action='store_true'
  )
  parser.add_argument(
      '--no_r2',
      default=False,
      help='If true, then R square will not be included as a metric for hyperparameter searching.',
      action='store_true'
  )
  parser.add_argument(
      '--split', type=str, default='random', help='Which splitter to use.'
  )
  parser.add_argument(
      '--threshold',
      type=float,
      nargs='*',
      default=[7.0],
      help='The threshold used in the mode named reg-threshold.'
  )
  parser.add_argument(
      '--plot',
      default=False,
      help='If true, then plots will be generated.',
      action='store_true'
  )
  '''
  parser.add_argument(
      '--direction',
      default=False,
      help='The direction of desired metric values. False for minimization, True\
        for maximization.',
      action='store_true'
  )
  '''
  parser.add_argument(
      '--out_path',
      type=str,
      default='.',
      help='The path of the repository that the output files should be stored.'
  )
  parser.add_argument(
      '--fold_num',
      type=int,
      default=5,
      help='Number of folds. Only useful when cross_validation is true.'
  )
  parser.add_argument(
      '--hyper_parameters', default=None, help='Dictionary of hyper parameters.'
  )
  parser.add_argument(
      '--hyper_param_search',
      default=False,
      help='Flag of whether hyperparameters will be searched.',
      action='store_true'
  )
  parser.add_argument(
      '--verbose_search',
      default=False,
      help='Flag of whether current best scores are outputted to a file in the \
        hyperparameter searching process.',
      action='store_true'
  )
  parser.add_argument(
      '--arithmetic_mean',
      default=False,
      help='Flag of whether arithmetic means are calculated for validation scores in \
        the hyperparameter searching process.',
      action='store_true'
  )
  parser.add_argument(
      '--max_iter',
      type=int,
      default=42,
      help='Maximum number of iterations for hyperparameter searching.'
  )
  parser.add_argument(
      '--search_range',
      type=int,
      default=3,
      help='Ratio of values to try in hyperparameter searching.'
  )
  parser.add_argument(
      '--reload',
      type=bool,
      default=True,
      help='Flag of whether datasets will be reloaded from existing ones or newly constructed.'
  )
  parser.add_argument(
      '--cross_validation',
      default=False,
      help='Flag of whether cross validations will be performed.',
      action='store_true'
  )
  parser.add_argument(
      '--aggregate',
      #nargs='*',
      action = 'append',
      #default=["davis_data/prot_desc.csv", "metz_data/prot_desc.csv"],
      help='A list of strings that tasks containing which should be aggregated based on the \
        same string. Say, we have tasks "toxcast_3000", "toxcast_3100", and the aggregate list\
        is ["toxcast", ...], then we should aggregate the two tasks into one composite "toxcast"\
        task.'      
  )
  parser.add_argument(
      '--aggregate_suffix_file',
      type=str,
      default=None,
      help='File name of the csv file storing the suffixes used for aggregating tasks.'
  )
  parser.add_argument(
      '--test',      
      default=False,
      help='Flag of whether there will be test data.',
      action='store_true'
  )
  parser.add_argument(
      '--predict_cold',      
      default=False,
      help='Flag of whether the split will leave "cold" entities in the test data.',
      action='store_true'
  )
  parser.add_argument(
      '--split_warm',      
      default=False,
      help='Flag of whether the split will not leave "cold" entities in the test data.',
      action='store_true'
  )
  parser.add_argument(
      '--filter_threshold',
      type=int,
      default=0,
      help='Threshold such that entities with observations no more than it would be filtered out.'
  )
  parser.add_argument(
      '--cold_drug',      
      default=False,
      help='Flag of whether the split will leave "cold" drugs in the test data.',
      action='store_true'
  )
  parser.add_argument(
      '--cold_target',      
      default=False,
      help='Flag of whether the split will leave "cold" targets in the test data.',
      action='store_true'
  )
  parser.add_argument(
      '--cold_drug_cluster',      
      default=False,
      help='Flag of whether the split will leave "cold cluster" drugs in the test data.',
      action='store_true'
  )
  parser.add_argument(
      '--early_stopping',      
      default=False,
      help='Flag of whether early stopping would be enabled. Does not apply to CV.',
      action='store_true'
  )
  parser.add_argument(
      '--evaluate_freq',
      type=int,
      default=3,
      help='If enable early stopping, the number of training epochs before evaluation.'
  )
  parser.add_argument(
      '--patience',
      type=int,
      default=3,
      help='In early stopping, the number of evaluations without improvement that \
        can be tolerated.'
  )
  parser.add_argument(
      '--seed', type=int, default=123, help='Random seed to be used in tensorflow.'
  )  
  parser.add_argument(
      '--log_file',
      type=str,
      default='GPhypersearch_temp.log',
      help='Name of the file storing the process of hyperparameter searching.'
  )  
  parser.add_argument(
      '--model_dir',
      type=str,
      default='./model_dir',
      help='Directory to store the log files in the training process. Can call\
        tensorboard with it as the logdir.'
  )
  parser.add_argument(
      '--prot_desc_path',
      #nargs='*',
      action = 'append',
      #default=["davis_data/prot_desc.csv", "metz_data/prot_desc.csv"],
      help='A list containing paths to protein descriptors.'      
  )
  parser.add_argument(
      '--intermediate_file',
      type=str,
      default='intermediate_cv.csv',
      help='File name of the csv file storing the results for separate CV folds.'
  )
  parser.add_argument(
      '--predict_only',      
      default=False,
      help='True if only predicting is performed, no training.',
      action='store_true'
  )
  parser.add_argument(
      '--restore_model',      
      default=False,
      help='True if restoring model stored in the model_dir. If you want to directly use the \
        trained model without any modification, you MUST set the "nb_epoch" parameter to 0. \
        Note that if --predict_only parameter is true, restore_model is set to true.',
      action='store_true'
  )
  parser.add_argument(
      '--tensorboard',      
      default=False,
      help='True if storing the checkpoints.',
      action='store_true'
  )
  parser.add_argument(
      '--csv_out',
      type=str,
      default='./NCI60_data/predictions.csv',
      help='File name of the csv file storing the prediction results.'
  )
  parser.add_argument(
      '--oversampled',      
      default=False,
      help='Indicating whether the oversampled dataset was used for training and/or \
        cross-validation.',
      action='store_true'
  )
  parser.add_argument(
      '--no_input_protein',      
      default=False,
      help='Indicating whether protein information is part of model input.',
      action='store_true'
  )
  parser.add_argument(
      '--weighted_metric_of_each_endpoint',      
      default=False,
      help="Indicating whether the 'aggregated' endpoints in multi-task models are treated \
        separately to calculate a weighted average of evaluation metrics by the number of \
        observations in each endpoint. If this is set to False, all 'aggregated' endpoints are \
        concatenated into one array, and the evaluation metrics are calculated on this \
        concatenated array. If the 'aggregated' endpoints are intrinsically different and cannot \
        be pooled together to calculate a metric, as in the case of the ToxCast dataset, it is \
        necessary to set it as True.",
      action='store_true'
  ) 
   
  parser.add_argument(
      '--remove_val_set_entries',      
      default=False,
      help='Indicating whether we remove the entries existing in the validation set from the \
        validation folds of the datasets.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  #pdb.set_trace()
  tf.app.run(main=run_analysis, argv=[sys.argv[0]] + unparsed)
