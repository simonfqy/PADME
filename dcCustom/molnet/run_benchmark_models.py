from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import os
import time
import sys
import pwd
import pdb
import csv
import deepchem
import dcCustom
from dcCustom.molnet.preset_hyper_parameters import hps

def model_regression(
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
            test=False,
            hyper_parameters=None,
            early_stopping = True,
            evaluate_freq = 3, # Number of training epochs before evaluating
            # for early stopping.
            patience = 3,
            model_dir="./model_dir",
            direction = None,
            seed=123,
            tensorboard = False):
  train_scores = {}
  valid_scores = {}
  test_scores = {}
  assert model in [
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
      'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression',
      'textcnn_regression', 'krr', 'ani', 'krr_ft', 'mpnn'
  ]
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model
  
  if model_name == 'weave_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_hidden = hyper_parameters['n_hidden']
    dropout_prob = hyper_parameters['dropout_prob']
    # n_pair_feat = hyper_parameters['n_pair_feat']
    # batch_size = 183
    # learning_rate = 4.8e-5
    # nb_epoch = 35
    # n_graph_feat = 371
    # n_hidden = 154
    # dropout_prob = 0.2

    model = dcCustom.models.WeaveTensorGraph(
      len(tasks),
      n_atom_feat=n_features,
      #n_pair_feat=n_pair_feat,
      n_hidden=n_hidden,
      n_graph_feat=n_graph_feat,
      batch_size=batch_size,
      learning_rate=learning_rate,
      use_queue=False,
      random_seed=seed,
      dropout_prob = dropout_prob,
      mode='regression',
      tensorboard = tensorboard,
      model_dir = model_dir,
      prot_desc_dict=prot_desc_dict,
      prot_desc_length=prot_desc_length)
  
  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  if nb_epoch is None:
    opt_epoch = model.fit(train_dataset, valid_dataset, restore=False, 
      metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers)
  else:
    opt_epoch = model.fit(train_dataset, valid_dataset, nb_epoch=nb_epoch, restore=False, 
      metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers)
  if not early_stopping:
    opt_epoch = None

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, valid_scores, test_scores, opt_epoch
  
def model_classification(
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
            test=False,
            hyper_parameters=None,
            early_stopping = True,
            evaluate_freq = 3, # Number of training epochs before evaluating
            # for early stopping.
            patience = 3,
            direction = None,
            seed=123,
            tensorboard = False,
            model_dir="./cls_model_dir"):
  train_scores = {}
  valid_scores = {}
  test_scores = {}
  assert model in [
      'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb',
      'weave', 'kernelsvm', 'textcnn', 'mpnn'
  ]
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model
  
  if model_name == 'weave':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_hidden = hyper_parameters['n_hidden']
    dropout_prob = hyper_parameters['dropout_prob']
    # n_pair_feat = hyper_parameters['n_pair_feat']
    # batch_size = 183
    # learning_rate = 4.8e-5
    # nb_epoch = 20
    # n_graph_feat = 371
    # n_hidden = 154
    # dropout_prob = 0.1

    model = dcCustom.models.WeaveTensorGraph(
      len(tasks),
      n_atom_feat=n_features,
      #n_pair_feat=n_pair_feat,
      #n_hidden=50,
      #n_graph_feat=128,
      n_hidden=n_hidden,
      n_graph_feat=n_graph_feat,
      batch_size=batch_size,
      learning_rate=learning_rate,
      use_queue=False,
      random_seed=seed,
      dropout_prob = dropout_prob,
      mode='classification',
      tensorboard = tensorboard,
      model_dir = model_dir,
      prot_desc_dict=prot_desc_dict,
      prot_desc_length=prot_desc_length)
  
  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  if nb_epoch is None:
    opt_epoch = model.fit(train_dataset, valid_dataset, restore=False, 
      metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers)
  else:
    opt_epoch = model.fit(train_dataset, valid_dataset, nb_epoch=nb_epoch, restore=False, 
      metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers)
  if not early_stopping:
    opt_epoch = None

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, valid_scores, test_scores, opt_epoch