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
            no_r2=False,
            no_concordance_index=False,
            direction = None,
            seed=123,
            tensorboard = True,
            plot=False,
            verbose_search=False,
            log_file=None,
            aggregated_tasks=[],
            predict_only=False,
            restore_model=False,
            prediction_file=None,
            input_protein=True):

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
  
  if model_name == 'graphconvreg':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']
    
    model = dcCustom.models.GraphConvModel(
        len(tasks),
        graph_conv_layers=[n_filters] * 2,
        dense_layer_size=n_fully_connected_nodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        dropout_prob=dropout_prob,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size = dense_cmb_layer_size,
        mode='regression',
        tensorboard=tensorboard,
        model_dir = model_dir,
        prot_desc_dict=prot_desc_dict,
        prot_desc_length=prot_desc_length,
        restore_model=restore_model,
        input_protein=input_protein)
        
  elif model_name == 'mpnn':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    T = hyper_parameters['T']
    M = hyper_parameters['M']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']

    model = dcCustom.models.MPNNModel(
        len(tasks),
        n_atom_feat=n_features[0],
        n_pair_feat=n_features[1],
        n_hidden=n_features[0],
        T=T,
        M=M,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        dropout_prob=dropout_prob,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size=dense_cmb_layer_size,
        mode="regression",
        tensorboard = tensorboard,
        model_dir = model_dir,
        prot_desc_dict = prot_desc_dict,
        prot_desc_length = prot_desc_length,
        restore_model=restore_model)
  
  elif model_name == 'tf_regression':
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    dropout_prob = hyper_parameters['dropout_prob']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']

    model = dcCustom.models.MultitaskRegressor(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        dropout_prob=dropout_prob,
        weight_decay_penalty=penalty,
        weight_decay_penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size = dense_cmb_layer_size,
        tensorboard = tensorboard,
        model_dir = model_dir,
        prot_desc_dict=prot_desc_dict,
        prot_desc_length=prot_desc_length,
        restore_model=restore_model,
        input_protein=input_protein)
  
  elif model_name == 'weave_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_hidden = hyper_parameters['n_hidden']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']
    # n_pair_feat = hyper_parameters['n_pair_feat']
    # batch_size = 177
    # learning_rate = 4.31e-5
    # nb_epoch = 1
    # n_graph_feat = 248
    # n_hidden = 202
    # dropout_prob = 0.153

    model = dcCustom.models.WeaveModel(
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
      num_dense_layer=num_dense_layer,
      dense_cmb_layer_size = dense_cmb_layer_size,
      mode='regression',
      tensorboard = tensorboard,
      model_dir = model_dir,
      prot_desc_dict=prot_desc_dict,
      prot_desc_length=prot_desc_length,
      restore_model=restore_model)
  
  if predict_only:
    assert prediction_file is not None
    model.predict(train_dataset, transformers=transformers, csv_out=prediction_file, tasks=tasks)
    return None, None, None, None

  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  
  if len(tasks) > 1:
    per_task_metrics = True
  else:
    per_task_metrics = False
  
  if nb_epoch is None:
    opt_epoch = model.fit(train_dataset, valid_dataset, restore=restore_model, 
      metric=metric, direction=direction, early_stopping=early_stopping, no_r2=no_r2,
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers,
      per_task_metrics=per_task_metrics, tasks=tasks, verbose_search=verbose_search,
      log_file=log_file, aggregated_tasks=aggregated_tasks, model_name=model_name)
  else:
    opt_epoch = model.fit(train_dataset, valid_dataset, nb_epoch=nb_epoch,
      restore=restore_model, metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers,
      per_task_metrics=per_task_metrics, tasks=tasks, verbose_search=verbose_search,
      log_file=log_file, aggregated_tasks=aggregated_tasks, model_name=model_name,
      no_r2=no_r2)
  if not early_stopping:
    opt_epoch = None
  
  if per_task_metrics:    
    train_sc, per_task_sc_train = model.evaluate(
      train_dataset, metric, transformers, per_task_metrics=per_task_metrics, 
      no_concordance_index=no_concordance_index, plot=plot, is_training_set=True, tasks=tasks,
      model_name=model_name)
    train_scores[model_name] = {'averaged': train_sc, 'per_task_score': per_task_sc_train}
    valid_sc, per_task_sc_val = model.evaluate(
      valid_dataset, metric, transformers, per_task_metrics=per_task_metrics, plot=plot,
      tasks=tasks, model_name=model_name)
    valid_scores[model_name] = {'averaged': valid_sc, 'per_task_score': per_task_sc_val}
    if test:
      test_sc, per_task_sc_test = model.evaluate(
        test_dataset, metric, transformers, per_task_metrics=per_task_metrics, 
        plot=plot, tasks=tasks, model_name=model_name)
      test_scores[model_name] = {'averaged': test_sc, 'per_task_score': per_task_sc_test}
      
  else:
    train_scores[model_name] = model.evaluate(train_dataset, metric, transformers, 
      per_task_metrics=per_task_metrics, no_concordance_index=no_concordance_index,
      plot=plot, is_training_set=True, tasks=tasks, model_name=model_name)
    valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers,
      per_task_metrics=per_task_metrics, plot=plot, tasks=tasks, model_name=model_name)
    if test:
      test_scores[model_name] = model.evaluate(test_dataset, metric, transformers,
        per_task_metrics=per_task_metrics, plot=plot, tasks=tasks, model_name=model_name)  

  return train_scores, valid_scores, test_scores, opt_epoch
  

# TODO: If you want to use this function, please update it according to model_regression()
# function in every detail.
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
            no_concordance_index=False,
            seed=123,
            tensorboard = True,
            model_dir="./cls_model_dir",
            plot=False,
            verbose_search=False,
            log_file=None,            
            aggregated_tasks=[],
            predict_only=False,
            restore_model=False,
            prediction_file=None,
            input_protein=True):
  
  train_scores = {}
  valid_scores = {}
  test_scores = {}
  per_task_train_scores = {}
  per_task_valid_scores = {}
  per_task_test_scores = {}
  
  assert model in [
      'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb',
      'weave', 'kernelsvm', 'textcnn', 'mpnn'
  ]
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model
  
  if model_name == 'graphconv':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']
    
    model = dcCustom.models.GraphConvModel(
        len(tasks),
        graph_conv_layers=[n_filters] * 2,
        dense_layer_size=n_fully_connected_nodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        dropout_prob=dropout_prob,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size = dense_cmb_layer_size,
        mode='classification',
        tensorboard=tensorboard,
        model_dir = model_dir,
        prot_desc_dict=prot_desc_dict,
        prot_desc_length=prot_desc_length,
        restore_model=predict_only)

  elif model_name == 'mpnn':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    T = hyper_parameters['T']
    M = hyper_parameters['M']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']

    model = dcCustom.models.MPNNModel(
        len(tasks),
        n_atom_feat=n_features[0],
        n_pair_feat=n_features[1],
        n_hidden=n_features[0],
        T=T,
        M=M,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        dropout_prob=dropout_prob,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size=dense_cmb_layer_size,
        mode="classification",
        tensorboard = tensorboard,
        model_dir = model_dir,
        prot_desc_dict = prot_desc_dict,
        prot_desc_length = prot_desc_length,
        restore_model=predict_only)

  elif model_name == 'tf':
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    dropout_prob = hyper_parameters['dropout_prob']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']

    # Building tensorflow MultitaskDNN model
    model = dcCustom.models.MultitaskClassifier(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        weight_decay_penalty=penalty,
        weight_decay_penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        num_dense_layer=num_dense_layer,
        dense_cmb_layer_size=dense_cmb_layer_size,
        tensorboard=tensorboard,
        model_dir=model_dir,
        prot_desc_dict=prot_desc_dict,
        prot_desc_length=prot_desc_length,
        restore_model=predict_only)
  
  elif model_name == 'weave':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_hidden = hyper_parameters['n_hidden']
    dropout_prob = hyper_parameters['dropout_prob']
    num_dense_layer = hyper_parameters['num_dense_layer']
    dense_cmb_layer_size = hyper_parameters['dense_cmb_layer_size']
    # n_pair_feat = hyper_parameters['n_pair_feat']
    # batch_size = 183
    # learning_rate = 4.8e-5
    # nb_epoch = 20
    # n_graph_feat = 371
    # n_hidden = 154
    # dropout_prob = 0.1

    model = dcCustom.models.WeaveModel(
      len(tasks),
      n_atom_feat=n_features,
      #n_pair_feat=n_pair_feat,      
      #n_graph_feat=128,
      n_hidden=n_hidden,
      n_graph_feat=n_graph_feat,
      batch_size=batch_size,
      learning_rate=learning_rate,
      use_queue=False,
      random_seed=seed,
      dropout_prob = dropout_prob,
      num_dense_layer=num_dense_layer,
      dense_cmb_layer_size=dense_cmb_layer_size,
      mode='classification',
      tensorboard = tensorboard,
      model_dir = model_dir,
      prot_desc_dict=prot_desc_dict,
      prot_desc_length=prot_desc_length,
      restore_model=predict_only)
  
  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  if len(tasks) > 1:
    per_task_metrics = True
  else:
    per_task_metrics = False
  
  if nb_epoch is None:
    opt_epoch = model.fit(train_dataset, valid_dataset, restore=False, 
      metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers,
      per_task_metrics=per_task_metrics, tasks=tasks)
  else:
    opt_epoch = model.fit(train_dataset, valid_dataset, nb_epoch=nb_epoch,
      restore=False, metric=metric, direction=direction, early_stopping=early_stopping, 
      evaluate_freq=evaluate_freq, patience = patience, transformers=transformers,
      per_task_metrics=per_task_metrics, tasks=tasks)
  if not early_stopping:
    opt_epoch = None
  
  if per_task_metrics:    
    train_sc, per_task_sc_train = model.evaluate(
      train_dataset, metric, transformers, per_task_metrics=per_task_metrics, 
      no_concordance_index=no_concordance_index)
    train_scores[model_name] = {'averaged': train_sc, 'per_task_score': per_task_sc_train}
    valid_sc, per_task_sc_val = model.evaluate(
      valid_dataset, metric, transformers, per_task_metrics=per_task_metrics)
    valid_scores[model_name] = {'averaged': valid_sc, 'per_task_score': per_task_sc_val}
    if test:
      test_sc, per_task_sc_test = model.evaluate(
        test_dataset, metric, transformers, per_task_metrics=per_task_metrics)
      test_scores[model_name] = {'averaged': test_sc, 'per_task_score': per_task_sc_test}
      
  else:
    train_scores[model_name] = model.evaluate(train_dataset, metric, transformers, 
      per_task_metrics=per_task_metrics, no_concordance_index=no_concordance_index)
    valid_scores[model_name] = model.evaluate(valid_dataset, metric, transformers,
      per_task_metrics=per_task_metrics)
    if test:
      test_scores[model_name] = model.evaluate(test_dataset, metric, transformers,
        per_task_metrics=per_task_metrics)  

  return train_scores, valid_scores, test_scores, opt_epoch
  