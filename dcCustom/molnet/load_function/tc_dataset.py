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
import deepchem
import pickle
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification
from dcCustom.molnet.check_availability import CheckFeaturizer, CheckSplit

def load_toxcast(featurizer = 'Weave', cross_validation=False, test=False, split='random', 
  reload=True, K = 5, mode = 'regression', predict_cold = False, cold_drug=False, 
  cold_target=False, cold_drug_cluster=False, split_warm=False, filter_threshold=0, 
  prot_seq_dict=None, currdir="./", oversampled=False, input_protein=True,
  remove_val_set_entries=False): 
  # The last parameter means only splitting into training and validation sets.

  if cross_validation:
    assert not test
  data_dir = currdir + "full_toxcast/"  
  if input_protein:
    if mode == 'regression' or mode == 'reg-threshold':
      mode = 'regression'      
      file_name = "restructured.csv"
    elif mode == 'classification':      
      file_name = "restructured_bin.csv"
    dataset_file = os.path.join(data_dir, file_name)
    df = pd.read_csv(dataset_file, header = 0, index_col=False)
    headers = list(df)
    tasks=headers[:-3]
  else:    
    if mode == 'regression' or mode == 'reg-threshold':
      mode = 'regression'
      file_name = "restructured_no_prot.csv"
    elif mode == 'classification':
      file_name = "restructured_bin_no_prot.csv"
    dataset_file = os.path.join(data_dir, file_name)
    df = pd.read_csv(dataset_file, header = 0, index_col=False)
    headers = list(df)    
    tasks = headers[:-1]

  
  if reload:
    delim = "_5/"
    if remove_val_set_entries:
      delim = "_rmval" + delim
    if not input_protein:
      delim = "_no_prot" + delim
    if filter_threshold > 0:
      delim = "_filtered" + delim
    if predict_cold:
      delim = "_cold" + delim
    elif split_warm:
      delim = "_warm" + delim
    elif cold_drug:
      delim = "_cold_drug" + delim
    elif cold_target:
      delim = "_cold_target" + delim
    elif cold_drug_cluster:
      delim = '_cold_drug_cluster' + delim
    if oversampled:
      delim = "_oversp" + delim
    if cross_validation:
      delim = "_CV" + delim
      save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
      loaded, all_dataset, transformers = dcCustom.utils.save.load_cv_dataset_from_disk(
          save_dir, K)
    else:
      save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
      loaded, all_dataset, transformers = dcCustom.utils.save.load_dataset_from_disk(
          save_dir)
    if loaded:
      return tasks, all_dataset, transformers 

  if remove_val_set_entries:
    if input_protein:
      save_dir_val_set = os.path.join(data_dir, featurizer + "/" + mode + "/" + split)
    else:      
      save_dir_val_set = os.path.join(data_dir, featurizer + "_cold_drug_no_prot/" + mode + "/" + split)
  else:
    save_dir_val_set = None
  
  if featurizer == 'Weave':
    featurizer = dcCustom.feat.WeaveFeaturizer()
  elif featurizer == 'ECFP':
    featurizer = dcCustom.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dcCustom.feat.ConvMolFeaturizer()
  
  loader = dcCustom.data.CSVLoader(
      tasks = tasks, smiles_field="smiles", protein_field = "proteinName", 
      source_field = 'protein_dataset', featurizer=featurizer, prot_seq_dict=prot_seq_dict,
      input_protein = input_protein)
  dataset = loader.featurize(dataset_file, shard_size=8192)
  
  if mode == 'regression':
    transformers = [
          dcCustom.trans.NormalizationTransformer(
              transform_y=True, dataset=dataset)
    ]
  elif mode == 'classification':
    transformers = [
        dcCustom.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]
    
  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)
    
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': dcCustom.splits.RandomSplitter(split_cold=predict_cold, cold_drug=cold_drug, 
        cold_target=cold_target, cold_drug_cluster=cold_drug_cluster, split_warm=split_warm, 
        prot_seq_dict=prot_seq_dict, threshold=filter_threshold, oversampled=oversampled, 
        input_protein=input_protein, remove_val_set_entries=remove_val_set_entries, 
        save_dir_val_set=save_dir_val_set),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'task': deepchem.splits.TaskSplitter()
  }
  splitter = splitters[split]
  if test:
    train, valid, test = splitter.train_valid_test_split(dataset)
    all_dataset = (train, valid, test)
    if reload:
      dcCustom.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)
  elif cross_validation:
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
    if reload:
      dcCustom.utils.save.save_cv_dataset_to_disk(save_dir, all_dataset, K, transformers)

  else:
    # not cross validating, and not testing.
    train, valid, test = splitter.train_valid_test_split(dataset, frac_train=0.9, frac_valid=0.1,
      frac_test=0)
    all_dataset = (train, valid, test)
    if reload:
      dcCustom.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                               transformers)
  
  return tasks, all_dataset, transformers  