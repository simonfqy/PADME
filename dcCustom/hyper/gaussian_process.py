"""
Contains class for gaussian process hyperparameter optimizations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tempfile
import os
import pdb
from deepchem.hyper.grid_search import HyperparamOpt
from deepchem.utils.evaluate import Evaluator
from dcCustom.molnet.run_benchmark_models import model_regression, model_classification
#from deepchem.molnet.run_benchmark_models import benchmark_classification


class GaussianProcessHyperparamOpt(HyperparamOpt):
  """
  Gaussian Process Global Optimization(GPGO)
  """

  def hyperparam_search(
      self,
      params_dict,
      train_dataset,
      valid_dataset,
      output_transformers,
      metric,
      prot_desc_dict,
      prot_desc_length,
      tasks=None,
      direction=True,
      n_features=1024,
      n_tasks=1,
      max_iter=20,
      search_range=4,
      early_stopping=True,
      evaluate_freq=3,
      patience=3,
      model_dir="./model_dir",
      hp_invalid_list=[
          'seed', 'nb_epoch', 'penalty_type', 'dropouts', 'bypass_dropouts',
          'n_pair_feat', 'fit_transformers', 'min_child_weight', 'weight_init_stddevs',
          'max_delta_step', 'subsample', 'colsample_bylevel', 'bias_init_consts',
          'colsample_bytree', 'reg_alpha', 'reg_lambda', 'scale_pos_weight',
          'base_score', 'layer_sizes'
      ],
      log_file='GPhypersearch.log',
      mode='classification',
      tensorboard=True,
      no_concordance_index=False,
      no_r2=False,
      plot=False,
      verbose_search=False,
      aggregated_tasks=[]):
    """Perform hyperparams search using a gaussian process assumption

    params_dict include single-valued parameters being optimized,
    which should only contain int, float and list of int(float)

    parameters with names in hp_invalid_list will not be changed.

    For Molnet models, self.model_class is model name in string,
    params_dict = dc.molnet.preset_hyper_parameters.hps[self.model_class]

    Parameters
    ----------
    params_dict: dict
      dict including parameters and their initial values
      parameters not suitable for optimization can be added to hp_invalid_list
    train_dataset: dc.data.Dataset struct
      dataset used for training
    valid_dataset: dc.data.Dataset struct
      dataset used for validation(optimization on valid scores)
    output_transformers: list of dc.trans.Transformer
      transformers for evaluation
    metric: list of dc.metrics.Metric
      metric used for evaluation
    direction: bool
      maximization(True) or minimization(False)
    n_features: int
      number of input features
    n_tasks: int
      number of tasks
    max_iter: int
      number of optimization trials
    search_range: int(float)
      optimization on [initial values / search_range,
                       initial values * search_range]
    hp_invalid_list: list
      names of parameters that should not be optimized
    logfile: string
      name of log file, hyperparameters and results for each trial will be recorded

    Returns
    -------
    hyper_parameters: dict
      params_dict with all optimized values
    valid_performance_opt: float
      best performance on valid dataset

    """

    #assert len(metric) == 1, 'Only use one metric'
    hyper_parameters = params_dict
    hp_list = list(hyper_parameters.keys())
    for hp in hp_invalid_list:
      if hp in hp_list:
        hp_list.remove(hp)

    hp_list_class = [hyper_parameters[hp].__class__ for hp in hp_list]
    assert set(hp_list_class) <= set([list, int, float])
    # Float or int hyper parameters(ex. batch_size, learning_rate)
    hp_list_single = [
        hp_list[i] for i in range(len(hp_list)) if not hp_list_class[i] is list
    ]
    # List of float or int hyper parameters(ex. layer_sizes)
    hp_list_multiple = [(hp_list[i], len(hyper_parameters[hp_list[i]]))
                        for i in range(len(hp_list))
                        if hp_list_class[i] is list]

    # Number of parameters
    n_param = len(hp_list_single)
    if len(hp_list_multiple) > 0:
      n_param = n_param + sum([hp[1] for hp in hp_list_multiple])
    # Range of optimization
    param_range = []
    for hp in hp_list_single:
      if hyper_parameters[hp].__class__ is int:
        param_range.append((('int'), [
            hyper_parameters[hp] // search_range,
            hyper_parameters[hp] * search_range
        ]))
      else:
        param_range.append((('cont'), [
            hyper_parameters[hp] / search_range,
            hyper_parameters[hp] * search_range
        ]))
    for hp in hp_list_multiple:
      if hyper_parameters[hp[0]][0].__class__ is int:
        param_range.extend([(('int'), [
            hyper_parameters[hp[0]][i] // search_range,
            hyper_parameters[hp[0]][i] * search_range
        ]) for i in range(hp[1])])
      else:
        param_range.extend([(('cont'), [
            hyper_parameters[hp[0]][i] / search_range,
            hyper_parameters[hp[0]][i] * search_range
        ]) for i in range(hp[1])])

    # Dummy names
    param_name = ['l' + format(i, '02d') for i in range(20)]
    param = dict(zip(param_name[:n_param], param_range))

    data_dir = './logs'
    log_file = os.path.join(data_dir, log_file)
        
    def f(l00=0,
          l01=0,
          l02=0,
          l03=0,
          l04=0,
          l05=0,
          l06=0,
          l07=0,
          l08=0,
          l09=0,
          l10=0,
          l11=0,
          l12=0,
          l13=0,
          l14=0,
          l15=0,
          l16=0,
          l17=0,
          l18=0,
          l19=0):
      """ Optimizing function
      Take in hyper parameter values and return valid set performances

      Parameters
      ----------
      l00~l19: int or float
        placeholders for hyperparameters being optimized,
        hyper_parameters dict is rebuilt based on input values of placeholders

      Returns:
      --------
      valid_scores: float
        valid set performances
      """
      args = locals()
      # Input hyper parameters
      i = 0
      for hp in hp_list_single:
        hyper_parameters[hp] = float(args[param_name[i]])
        if param_range[i][0] == 'int':
          hyper_parameters[hp] = int(hyper_parameters[hp])
        i = i + 1
      for hp in hp_list_multiple:
        hyper_parameters[hp[0]] = [
            float(args[param_name[j]]) for j in range(i, i + hp[1])
        ]
        if param_range[i][0] == 'int':          
          hyper_parameters[hp[0]] = list(map(int, hyper_parameters[hp[0]]))
        i = i + hp[1]
      
      opt_epoch = -1
      print(hyper_parameters)
      nonlocal model_dir
      
      pdb.set_trace()
      # Run benchmark
      with open(log_file, 'a') as f:
        # Record hyperparameters
        f.write(str(hyper_parameters))
        f.write('\n')
      if isinstance(self.model_class, str) or isinstance(
          self.model_class, unicode):
        
        if mode == 'classification':          
          train_scores, valid_scores, _, opt_epoch = model_classification(
              train_dataset,
              valid_dataset,
              valid_dataset, 
              tasks,
              output_transformers,
              n_features,
              metric,
              self.model_class,
              prot_desc_dict,
              prot_desc_length,
              hyper_parameters=hyper_parameters,
              early_stopping=early_stopping,
              evaluate_freq=evaluate_freq,
              patience=patience,
              direction=direction,
              model_dir=model_dir,
              tensorboard=tensorboard,
              no_concordance_index=no_concordance_index,
              verbose_search=verbose_search,
              log_file=log_file,
              no_r2=no_r2,
              aggregated_tasks=aggregated_tasks)
        elif mode == 'regression' or mode == 'reg-threshold':          
          train_scores, valid_scores, _, opt_epoch = model_regression(
              train_dataset,
              valid_dataset,
              valid_dataset, 
              tasks,
              output_transformers,
              n_features,
              metric,
              self.model_class,
              prot_desc_dict,
              prot_desc_length,
              hyper_parameters=hyper_parameters,
              early_stopping=early_stopping,
              evaluate_freq=evaluate_freq,
              patience=patience,
              direction=direction,
              model_dir=model_dir,
              tensorboard=tensorboard,
              no_concordance_index=no_concordance_index,
              verbose_search=verbose_search,
              log_file=log_file,
              no_r2=no_r2,
              aggregated_tasks=aggregated_tasks)
        else:
          raise ValueError("Invalid mode!")
        # similar to fit() function in tensor_graph.py, we also use combination here.
        if n_tasks > 1:
          val_scores = valid_scores[self.model_class]['averaged']
        else:
          val_scores = valid_scores[self.model_class]
        score = 0
        if mode == 'regression':
          for mtc in metric:
            mtc_name = mtc.metric.__name__
            composite_mtc_name = mtc.name
            if mtc_name == 'rms_score':
              score += val_scores[composite_mtc_name]
            if mtc_name == 'r2_score' or mtc_name == 'pearson_r2_score':              
              if no_r2:
                coef = 0.0
              else:
                coef = -0.5
              score += coef * val_scores[composite_mtc_name]
            if mtc_name == 'concordance_index':
              score += -val_scores[composite_mtc_name]
        elif mode == 'reg-threshold' or mode == 'classification':
          for mtc in metric:
            mtc_name = mtc.metric.__name__
            composite_mtc_name = mtc.name
            if mtc_name == 'roc_auc_score':
              score += val_scores[composite_mtc_name]            
            if mtc_name == 'prc_auc_score':
              score += val_scores[composite_mtc_name]

      else:
        model_dir = tempfile.mkdtemp()
        model = self.model_class(hyper_parameters, model_dir)
        model.fit(train_dataset, **hyper_parameters)
        model.save()
        evaluator = Evaluator(model, valid_dataset, output_transformers)
        multitask_scores = evaluator.compute_model_performance([metric])
        score = multitask_scores[metric.name]
      
      #pdb.set_trace()
      if early_stopping:
        best_score = opt_epoch[1]
        opt_epoch = opt_epoch[0]
      epoch_stmt = str(opt_epoch) + " is the optimum number of epochs found."
      print(epoch_stmt)
      with open(log_file, 'a') as f:
        # Record performances
        f.write(self.model_class)
        f.write('\n')
        f.write(epoch_stmt)
        f.write('\n')
        f.write(str(score))
        f.write('\n')
        if early_stopping:
          f.write(str(best_score))
          f.write('\n')
      if not early_stopping:
        best_score = score
      # GPGO maximize performance by default, set performance to its negative value for minimization
      if direction:
        return best_score
      else:
        return -best_score

    import pyGPGO
    from pyGPGO.covfunc import matern32
    from pyGPGO.acquisition import Acquisition
    from pyGPGO.surrogates.GaussianProcess import GaussianProcess
    from pyGPGO.GPGO import GPGO
    with open(log_file, 'a') as file:
      file.write("------------------------------------------------------------------")
      file.write('\n')
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, f, param)
    print("Max number of iteration: %i" % max_iter)
    
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()
    # Readout best hyper parameters
    i = 0
    for hp in hp_list_single:
      hyper_parameters[hp] = float(hp_opt[param_name[i]])
      if param_range[i][0] == 'int':
        hyper_parameters[hp] = int(hyper_parameters[hp])
      i = i + 1
    for hp in hp_list_multiple:
      hyper_parameters[hp[0]] = [
          float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])
      ]
      if param_range[i][0] == 'int':
        hyper_parameters[hp[0]] = list(map(int, hyper_parameters[hp[0]]))
      i = i + hp[1]
    
    opt_epoch = -1
    # Compare best model to default hyperparameters
    with open(log_file, 'a') as f:
      # Record hyperparameters
      f.write(str(params_dict))
      f.write('\n')
    if isinstance(self.model_class, str) or isinstance(self.model_class,
                                                       unicode):
      if mode == 'classification':
        train_scores, valid_scores, _, opt_epoch = model_classification(
            train_dataset,
            valid_dataset,
            valid_dataset, 
            tasks,
            output_transformers,
            n_features,
            metric,
            self.model_class,
            prot_desc_dict,
            prot_desc_length,           
            hyper_parameters=params_dict,
            early_stopping=early_stopping,
            evaluate_freq=evaluate_freq,
            patience=patience,
            direction=direction,
            model_dir=model_dir,
            tensorboard=tensorboard,
            no_concordance_index=no_concordance_index,
            verbose_search=verbose_search,
            log_file=log_file,
            no_r2=no_r2,
            aggregated_tasks=aggregated_tasks)
      elif mode == 'regression' or mode == 'reg-threshold':
        train_scores, valid_scores, _, opt_epoch = model_regression(
            train_dataset,
            valid_dataset,
            valid_dataset, 
            tasks,
            output_transformers,
            n_features,
            metric,
            self.model_class,
            prot_desc_dict,
            prot_desc_length,
            hyper_parameters=params_dict,
            early_stopping=early_stopping,
            evaluate_freq=evaluate_freq,
            patience=patience,
            direction=direction,
            model_dir=model_dir,
            tensorboard=tensorboard,
            no_concordance_index=no_concordance_index,
            verbose_search=verbose_search,
            log_file=log_file,
            no_r2=no_r2,
            aggregated_tasks=aggregated_tasks)
      else:
        raise ValueError("Invalid mode!")
      
      if n_tasks > 1:
        val_scores = valid_scores[self.model_class]['averaged']
      else:
        val_scores = valid_scores[self.model_class]
      score = 0
      if mode == 'regression':
        for mtc in metric:
          mtc_name = mtc.metric.__name__
          composite_mtc_name = mtc.name
          if mtc_name == 'rms_score':
            score += val_scores[composite_mtc_name]
          if mtc_name == 'r2_score' or mtc_name == 'pearson_r2_score':
            if no_r2:
              coef = 0.0
            else:
              coef = -0.5
            score += coef * val_scores[composite_mtc_name]
          if mtc_name == 'concordance_index':
            score += -val_scores[composite_mtc_name]
      elif mode == 'reg-threshold' or mode == 'classification':
        for mtc in metric:
          mtc_name = mtc.metric.__name__
          composite_mtc_name = mtc.name
          if mtc_name == 'roc_auc_score':
            score += val_scores[composite_mtc_name]            
          if mtc_name == 'prc_auc_score':
            score += val_scores[composite_mtc_name]
      
      if early_stopping:
        best_score = opt_epoch[1]
        opt_epoch = opt_epoch[0]
      
      epoch_stmt = str(opt_epoch) + " is the optimum number of epochs found."
      print(epoch_stmt)  
        
      #pdb.set_trace()
      with open(log_file, 'a') as f:
        f.write(epoch_stmt)
        f.write('\n')
        # Record performances
        f.write(str(score))
        f.write('\n')
        if early_stopping:
          f.write(str(best_score))
          f.write('\n')

      if not early_stopping:
        best_score = score
      
      # I have changed the determination criteria from score to best_score.
      if not direction:
        best_score = -best_score
      if best_score > valid_performance_opt:
        # Default model is better, return hyperparameters
        return params_dict, best_score

    # Return optimized hyperparameters
    return hyper_parameters, valid_performance_opt
