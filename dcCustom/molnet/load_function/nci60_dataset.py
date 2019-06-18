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

# This dataset is for prediction only, there is no true values known.
def load_nci60(featurizer = 'Weave', cross_validation=False, test=False, 
  split='random', reload=True, K = 5, mode = 'regression', predict_cold = False, 
  cold_drug=False, cold_target=False, split_warm=False, filter_threshold=0,
  prot_seq_dict=None, oversampled=False): 
  
  data_to_train = 'tc'
  
  if mode == 'regression' or mode == 'reg-threshold':
    mode = 'regression'
    file_name = "AR_ER_intxn_s"
    
  elif mode == 'classification':   
    file_name = "restructured_bin"

  file_name = file_name + "_" + data_to_train + '.csv'
  data_dir = "NCI60_data/"
  dataset_file = os.path.join(data_dir, file_name)
  df = pd.read_csv(dataset_file, header = 0, index_col=False)
  headers = list(df)
  tasks = headers[:-3]
  
  if reload:
    delim = "_" + data_to_train + "/"    
    save_dir = os.path.join(data_dir, featurizer + delim + mode + "/" + split)
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return tasks, all_dataset, transformers

  # HACK: the following if-else block could be prone to errors.
  if data_to_train == "tc":
    loaded, _, transformers = dcCustom.molnet.load_toxcast(featurizer = featurizer, split= split, 
      cross_validation=False, reload=True, mode = mode)
  else:
    # We assume there are only toxcast and kiba as the choice now.
    loaded, _, transformers = dcCustom.molnet.load_kiba(featurizer = featurizer, split= split, 
      cross_validation=False, reload=True, mode = mode, split_warm=True, filter_threshold=6)
  
  assert loaded
  
  if featurizer == 'Weave':
    featurizer = dcCustom.feat.WeaveFeaturizer()
  elif featurizer == 'ECFP':
    featurizer = dcCustom.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dcCustom.feat.ConvMolFeaturizer()
  
  loader = dcCustom.data.CSVLoader(
      tasks = tasks, smiles_field="smiles", protein_field = "proteinName", 
      source_field = 'protein_dataset', featurizer=featurizer, prot_seq_dict=prot_seq_dict)
  dataset = loader.featurize(dataset_file, shard_size=8192)  
      
  # print("About to transform data")
  # for transformer in transformers:
  #   dataset = transformer.transform(dataset)
    
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': dcCustom.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'task': deepchem.splits.TaskSplitter()
  }
  splitter = splitters[split]  
  
  # HACK: We set frac_train to 1.0 because assume NCI60 dataset is for prediction only: there
  # is no underlying truth. To predict all drug-target pairs, we need to let all samples be in
  # the "training" set, though it is a misnomer.
  train, valid, test = splitter.train_valid_test_split(dataset, frac_train=1.0, 
    frac_valid=0.0, frac_test=0)
  all_dataset = (train, valid, test)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  
  return tasks, all_dataset, transformers  