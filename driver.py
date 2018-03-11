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
import re
import deepchem
import pickle
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification
from dcCustom.molnet.check_availability import CheckFeaturizer, CheckSplit

FLAGS = None

def load_davis(featurizer = 'Weave', cross_validation=False, test=False, split='random', 
  reload=True, K = 5, mode = 'regression', predict_cold = False): 
  # The last parameter means only splitting into training and validation sets.

  if cross_validation:
    assert not test

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

def load_kinases(featurizer = 'Weave', cross_validation=False, test=False, split='random', 
  reload=True, K = 5, mode = 'regression', predict_cold = False): 
  # The last parameter means only splitting into training and validation sets.

  if cross_validation:
    assert not test

  if mode == 'regression' or mode == 'reg-threshold':
    mode = 'regression'
    tasks = ['davis', 'metz', 'kiba']
    file_name = "davis_metz_kiba.csv"
  elif mode == 'classification':
    tasks = ['davis_bin', 'metz_bin', 'kiba_bin']
    file_name = "davis_metz_kiba_bin.csv"

  data_dir = "synthesized_data/"
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
  elif featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  
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
  
def load_prot_desc_dict(prot_desc_dict, prot_desc_path):
  df = pd.read_csv(prot_desc_path, index_col=0)
  #protList = list(df.index)  
  for row in df.itertuples():
    descriptor = row[2:]
    descriptor = np.array(descriptor)
    descriptor = np.reshape(descriptor, (1, len(descriptor)))
    assert row[0] not in prot_desc_dict
    prot_desc_dict[row[0]] = descriptor    

def run_analysis(_):
  dataset=FLAGS.dataset 
  model = FLAGS.model
  thresholding = FLAGS.thresholding
  split= FLAGS.split
  threshold = FLAGS.threshold
  direction = FLAGS.direction
  out_path = FLAGS.out_path
  fold_num = FLAGS.fold_num
  hyper_parameters=FLAGS.hyper_parameters
  hyper_param_search = FLAGS.hyper_param_search
  max_iter = FLAGS.max_iter
  search_range = FLAGS.search_range
  reload = FLAGS.reload
  cross_validation = FLAGS.cross_validation
  test = FLAGS.test
  predict_cold = FLAGS.predict_cold # Determines whether cold-start
  #drugs and targets are tested or validated.
  early_stopping = FLAGS.early_stopping
  evaluate_freq = FLAGS.evaluate_freq # Number of training epochs before evaluating
  # for early stopping.
  patience = FLAGS.patience
  seed=FLAGS.seed
  log_file = FLAGS.log_file
  model_dir = FLAGS.model_dir
  prot_desc_path=FLAGS.prot_desc_path                
                 
  assert model == model # Not a NAN
  searchObj = re.search('reg', model)
  mode = 'regression' if searchObj else 'classification'
  if mode == 'regression':
    if thresholding:
      mode = 'reg-threshold'

  if mode == 'regression':
    metric = [deepchem.metrics.Metric(deepchem.metrics.rms_score, np.mean)]
  elif mode == 'classification':
    metric = [deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean),
      deepchem.metrics.Metric(deepchem.metrics.prc_auc_score, np.mean)]
  elif mode == "reg-threshold":
    # TODO: this [0] is just a temporary solution. Need to implement per-task thresholds.
    # It is not a very trivial task.
    metric = [dcCustom.metrics.Metric(dcCustom.metrics.roc_auc_score, np.mean, 
      threshold=threshold[0], mode="regression")]

  # We assume that values above the threshold are "on" or 1, those below are "off"
  # or 0.  
  loading_functions = {
    'davis': load_davis,
    'all_kinase': load_kinases
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
  
  if cross_validation:    
    tasks, all_dataset, transformers = loading_functions[dataset](featurizer=featurizer, 
                                                  cross_validation=cross_validation,
                                                  test=test, split=split, reload=reload, 
                                                  K = fold_num, mode=mode, predict_cold=predict_cold)
  else:
    tasks, all_dataset, transformers = loading_functions[dataset](featurizer=featurizer, 
                                                  cross_validation=cross_validation,
                                                  test=test, split=split, reload=reload, mode=mode,
                                                  predict_cold=predict_cold)
  prot_desc_dict = {}
  for path in prot_desc_path:
    load_prot_desc_dict(prot_desc_dict, path)
  prot_desc_length = 8421
  #pdb.set_trace()
  
  # all_dataset will be a list of 5 elements (since we will use 5-fold cross validation),
  # each element is a tuple, in which the first entry is a training dataset, the second is
  # a validation dataset.

  time_start_fitting = time.time()
  train_scores_list = []
  valid_scores_list = []
  test_scores_list = []
    
  # TODO: don't forget to check the gaussian_process.py. There might be problems.
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
        log_file=log_file)
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
      train_score, valid_score, _, _ = model_functions[mode](
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
      # TODO: I made the decision to force disable early stopping for cross validation here,
      # not quite sure whether this is right.
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

        train_score_dict = train_score[model_name]
        valid_score_dict = valid_score[model_name]
        
        if len(tasks > 1):
          train_score_dict = train_score[model_name]['averaged']
          valid_score_dict = valid_score[model_name]['averaged']          

        for i in train_score_dict:
          # i here is the metric name, like 'rms_score'.
          if len(tasks) > 1:
            this_train_score = train_score_dict[i]
            this_valid_score = valid_score_dict[i]            
          else:
            this_train_score = train_score_dict[i]
            this_valid_score = valid_score_dict[i]            
          
          output_line = [
                dataset,
                model_name, i, 'train',
                this_train_score, 'valid', this_valid_score
          ]          
          output_line.extend(
              ['time_for_running', time_finish_fitting - time_start_fitting])
          writer.writerow(output_line)
          
          if len(tasks) > 1:
            train_score_tasks = train_score[model_name]['per_task_score'][i]
            valid_score_tasks = valid_score[model_name]['per_task_score'][i]
            
            for index, task in enumerate(tasks):
              dataset_nm = dataset + '_' + task
              output_line = [
                      dataset_nm,
                      model_name, i, 'train',
                      train_score_tasks[index], 'valid', valid_score_tasks[index]
              ]              
              writer.writerow(output_line)  
    else:
      train_score = train_scores_list[0]
      valid_score = valid_scores_list[0]
      if test:
        test_score = test_scores_list[0]

      train_score_dict = train_score[model_name]
      valid_score_dict = valid_score[model_name]
      if test:
        test_score_dict = test_score[model_name]
      if len(tasks > 1):
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
        if early_stopping:
          output_line.extend(['optimal epoch', opt_epoch[0]])
          output_line.extend(['best validation score', opt_epoch[1]])
        writer.writerow(output_line)
        
        if len(tasks) > 1:
          train_score_tasks = train_score[model_name]['per_task_score'][i]
          valid_score_tasks = valid_score[model_name]['per_task_score'][i]
          if test:
            test_score_tasks = test_score[model_name]['per_task_score'][i]
          for index, task in enumerate(tasks):
            dataset_nm = dataset + '_' + task
            output_line = [
                    dataset_nm,
                    model_name, i, 'train',
                    train_score_tasks[index], 'valid', valid_score_tasks[index]
            ]
            if test:
              output_line.extend(['test', test_score_tasks[index]])            
            writer.writerow(output_line)
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
      '--direction',
      type=bool,
      default=False,
      help='The direction of desired metric values. False for minimization, True\
        for maximization.'
  )
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

  FLAGS, unparsed = parser.parse_known_args()
  #pdb.set_trace()
  tf.app.run(main=run_analysis, argv=[sys.argv[0]] + unparsed)
