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
import csv
import re
import deepchem
import pickle
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification
from dcCustom.molnet.check_availability import CheckFeaturizer, CheckSplit

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