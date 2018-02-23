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
import pwd
import pdb
import csv
import deepchem
import pickle
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification


def load_davis(featurizer = 'Weave', cross_validation=False, test=False, split='random', 
  reload=True, K = 5, mode = 'regression', train_valid_only = True, predict_cold = False): 
  # The last parameter means only splitting into training and validation sets.

  if cross_validation:
    assert not test
    assert train_valid_only

  train_valid_only = not test

  if mode == 'regression' or mode == 'reg-threshold':
    mode = 'regression'
    tasks = ['interaction_value']
    file_name = "restructured.csv"
  elif mode == 'classification':
    tasks = ['interaction_bin']
    file_name = "restructured_bin.csv"

  data_dir = "davis_data/"
  if reload:
    delim = "/"
    if predict_cold:
      delim = "_cold" + delim
    if cross_validation:
      delim = "_CV" + delim
      save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
      loaded, all_dataset, transformers = dcCustom.utils.save.load_cv_dataset_from_disk(
          save_dir, K)
    else:
      save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
      loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
          save_dir)
    if loaded:
      return tasks, all_dataset, transformers
  
  dataset_file = os.path.join(data_dir, file_name)
  if featurizer == 'Weave':
    featurizer = dcCustom.feat.WeaveFeaturizer()
  loader = dcCustom.data.CSVLoader(
      tasks = tasks, smiles_field="smiles", protein_field = "proteinName",
      featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)
  
  if mode == 'regression':
    transformers = [
          deepchem.trans.NormalizationTransformer(
              transform_y=True, dataset=dataset)
    ]
  elif mode == 'classification':
    transformers = [
        deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]
    
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)
    
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': dcCustom.splits.RandomSplitter(split_cold=predict_cold),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'task': deepchem.splits.TaskSplitter()
  }
  splitter = splitters[split]
  if test:
    train, valid, test = splitter.train_valid_test_split(dataset)
    all_dataset = (train, valid, test)
    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)
  elif cross_validation:
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
    if reload:
      dcCustom.utils.save.save_cv_dataset_to_disk(save_dir, all_dataset, K, transformers)

  else:
    # not cross validating, and not testing.
    train, valid, test = splitter.train_valid_test_split(dataset, frac_valid=0.2,
      frac_test=0)
    all_dataset = (train, valid, test)
    if reload:
      deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)
  
  return tasks, all_dataset, transformers
  # TODO: the implementation above could be prone to errors. Not entirely sure.

def load_prot_desc_dict(prot_desc_path):
  df = pd.read_csv(prot_desc_path, index_col=0)
  #protList = list(df.index)
  prot_desc_dict = {}
  for row in df.itertuples():
    descriptor = row[2:]
    descriptor = np.array(descriptor)
    descriptor = np.reshape(descriptor, (1, len(descriptor)))
    prot_desc_dict[row[0]] = descriptor    
  return prot_desc_dict

def run_analysis(dataset='davis', 
                 featurizer = 'Weave',
                 mode = 'classification',
                 split= 'random',
                 threshold = 7.0,
                 direction = False,
                 out_path = '.',
                 fold_num = 5,
                 hyper_parameters=None,
                 hyper_param_search = False, 
                 max_iter = 42,
                 search_range = 3,
                 reload = True,
                 cross_validation = True,
                 test = False,
                 predict_cold = False, # Determines whether cold-start drugs and targets are tested or validated.
                 early_stopping = True,
                 evaluate_freq = 3, # Number of training epochs before evaluating
                 # for early stopping.
                 patience = 3,
                 seed=123,
                 log_file = 'GPhypersearch_temp.log',
                 model_dir = './cls_model_dir',
                 prot_desc_path="davis_data/prot_desc.csv"):
  if mode == 'regression':
    metric = [deepchem.metrics.Metric(deepchem.metrics.rms_score, np.mean)]
  elif mode == 'classification':
    metric = [deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean)]
  elif mode == "reg-threshold":
    metric = [dcCustom.metrics.Metric(dcCustom.metrics.roc_auc_score, np.mean, 
      threshold=threshold, mode="regression")]

  # We assume that values above the threshold are "on" or 1, those below are "off"
  # or 0.

  print('-------------------------------------')
  print('Running on dataset: %s' % dataset)
  print('-------------------------------------')
  
  if cross_validation:    
    tasks, all_dataset, transformers = load_davis(featurizer=featurizer, cross_validation=cross_validation,
                                                  test=test, split=split, reload=reload, 
                                                  K = fold_num, mode=mode, predict_cold=predict_cold)
  else:
    tasks, all_dataset, transformers = load_davis(featurizer=featurizer, cross_validation=cross_validation,
                                                  test=test, split=split, reload=reload, mode=mode,
                                                  predict_cold=predict_cold)
    
  prot_desc_dict = load_prot_desc_dict(prot_desc_path)
  prot_desc_length = 8421
  
  # all_dataset will be a list of 5 elements (since we will use 5-fold cross validation),
  # each element is a tuple, in which the first entry is a training dataset, the second is
  # a validation dataset.

  time_start_fitting = time.time()
  train_scores_list = []
  valid_scores_list = []
  test_scores_list = []
  if mode == 'regression' or mode == 'reg-threshold':
    model = 'weave_regression'
  elif mode == 'classification':
    model = 'weave'
    
  n_features = 75
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
        metric,
        prot_desc_dict,
        prot_desc_length,
        direction=direction,
        n_features=n_features,
        n_tasks=len(tasks),
        max_iter=max_iter,
        search_range=search_range,
        early_stopping = early_stopping,
        evaluate_freq=evaluate_freq,
        patience=patience,
        model_dir=model_dir,
        log_file=log_file)
    hyper_parameters = hyper_param_opt
  
  opt_epoch = -1
  test_dataset = None
  if mode == 'regression' or mode == 'reg-threshold':
    if not cross_validation:
      train_dataset, valid_dataset, test_dataset = all_dataset
      train_score, valid_score, test_score, opt_epoch = model_regression(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
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
            model_dir=model_dir)
      train_scores_list.append(train_score)
      valid_scores_list.append(valid_score)
      test_scores_list.append(test_score)
    else:
      for i in range(fold_num):
        train_score, valid_score, _, _ = model_regression(
            all_dataset[i][0],
            all_dataset[i][1],
            None,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            prot_desc_dict,
            prot_desc_length,
            hyper_parameters=hyper_parameters,
            test = test,
            early_stopping = False,
            direction=direction,
            seed=seed,
            model_dir=model_dir)

        train_scores_list.append(train_score)
        valid_scores_list.append(valid_score)
 
  elif mode == 'classification':
    direction=True
    if not cross_validation:
      train_dataset, valid_dataset, test_dataset = all_dataset
      train_score, valid_score, test_score, opt_epoch = model_classification(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
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
            model_dir=model_dir)
      train_scores_list.append(train_score)
      valid_scores_list.append(valid_score)
      test_scores_list.append(test_score)
    else:
      for i in range(fold_num):
        train_score, valid_score, _, _ = model_classification(
            all_dataset[i][0],
            all_dataset[i][1],
            None,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            prot_desc_dict,
            prot_desc_length,
            hyper_parameters=hyper_parameters,
            test = test,
            early_stopping = False,
            direction=direction,
            seed=seed,
            model_dir=model_dir)

        train_scores_list.append(train_score)
        valid_scores_list.append(valid_score)
  time_finish_fitting = time.time()
  
  results_file = 'results'
  
  if mode == 'classification':
    results_file += '_cls'
  elif mode == 'reg-threshold':
    results_file += '_thrhd'
  if predict_cold:
    results_file += '_cold'
  if cross_validation:
    results_file += '_cv'

  results_file += '.csv'

  with open(os.path.join(out_path, results_file), 'a') as f:
    writer = csv.writer(f)
    model_name = list(train_scores_list[0].keys())[0]
        
    if cross_validation:
      for h in range(fold_num):
        train_score = train_scores_list[h]
        valid_score = valid_scores_list[h]
        for i in train_score[model_name]:
          output_line = [
                dataset,
                model_name, i, 'train',
                train_score[model_name][i], 'valid', valid_score[model_name][i]
          ]          
          output_line.extend(
              ['time_for_running', time_finish_fitting - time_start_fitting])
          writer.writerow(output_line)
    else:
      train_score = train_scores_list[0]
      valid_score = valid_scores_list[0]
      if test:
        test_score = test_scores_list[0]
      for i in train_score[model_name]:
        output_line = [
                  dataset,
                  model_name, i, 'train',
                  train_score[model_name][i], 'valid', valid_score[model_name][i]
        ]
        if test:
          output_line.extend(['test', test_score[model_name][i]])
        if early_stopping:
          output_line.extend(['optimal epoch', opt_epoch])
        writer.writerow(output_line)
  if hyper_param_search:
    with open(os.path.join(out_path, dataset + model + '.pkl'), 'w') as f:
      pickle.dump(hyper_parameters, f)
  
if __name__ == '__main__':
  run_analysis()


