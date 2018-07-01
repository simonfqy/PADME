from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os
import sys
import pwd
import pdb
import csv
import re
import math
import matplotlib.pyplot as plt
import time
import dcCustom
from dcCustom.feat import Protein

AR_list = [('toxcast', 'P10275'), ('toxcast', 'O97775'), ('toxcast', 'P15207')]
AR_list_s = [('toxcast', 'P10275')]
ER_list = [('toxcast', 'P03372'), ('toxcast', 'P19785'), ('toxcast', 'P49884'), 
  ('toxcast', 'Q92731')]
ER_list_s = [('toxcast', 'P03372')]
AR_toxcast_codes = ['6620', '8110', '8010', '7100', '7200', '3100', '7300', '8100', '8000']
AR_antagonist_score_coef = [1, 0.5, 0.5, -0.5, -0.5, -1, -1/3, -1/3, -1/3]

PROT_desc_path_list = ['../davis_data/prot_desc.csv', '../metz_data/prot_desc.csv', 
    '../KIBA_data/prot_desc.csv', '../full_toxcast/prot_desc.csv']

def get_canonical_smiles_dict(file_name='invalid_smiles_canonicalized.csv'):
  smiles_file = pd.read_csv(file_name, header=0, index_col=False)
  invalid_smiles = smiles_file.iloc[:, 0]
  canonical_smiles = smiles_file.iloc[:, 1]
  smiles_dict = dict(zip(invalid_smiles, canonical_smiles))
  return smiles_dict
  
def get_smiles_from_prev(file_name_list=[]):
  complete_smiles_list = []
  for file_name in file_name_list:
    restructured = pd.read_csv(file_name, header=0, index_col=False)
    smiles_list = restructured.loc[:, 'smiles']
    smiles_list = list(set(smiles_list))
    complete_smiles_list = complete_smiles_list + smiles_list
  return complete_smiles_list
    
def load_prot_dict(protein_list, prot_desc_path, sequence_field, 
  phospho_field, pairs_to_choose=[]):
  if re.search('davis', prot_desc_path, re.I):
    source = 'davis'
  elif re.search('metz', prot_desc_path, re.I):
    source = 'metz'
  elif re.search('kiba', prot_desc_path, re.I):
    source = 'kiba'
  elif re.search('toxcast', prot_desc_path, re.I):
    source = 'toxcast'

  df = pd.read_csv(prot_desc_path, index_col=0)
  for row in df.itertuples():
    if len(pairs_to_choose) > 0:
      if len(protein_list) < len(pairs_to_choose):
        pair = (source, row[0])
        if pair not in pairs_to_choose:
          continue
      else:
        continue  

    sequence = row[sequence_field]
    phosphorylated = row[phospho_field]
    protein = Protein(row[0], source, (phosphorylated, sequence))
    if protein not in set(protein_list):
      protein_list.append(protein)  
    
def produce_dataset(dataset_used='toxcast', prot_desc_path_list=PROT_desc_path_list,
  get_all_compounds=False, take_mol_subset=True, mols_to_choose=2000, prot_pairs_to_choose=[], 
  output_prefix='AR_ER_intxn_s'):
  
  assert dataset_used in ['toxcast', 'kiba']
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  df = pd.read_csv('NCI60_bio.csv', header = 0, index_col=2)
  #df = df.head(60000)
  molList = list(df.index)
  molList = [mol for mol in molList if mol==mol]
  assert len(df) == len(molList)  
  selected_mol_set = set()
  selected_mol_list = []
  
  GIarray = np.asarray(df.iloc[:, 5])
  sorted_indices = np.argsort(GIarray)
  for i in sorted_indices:
    smiles = molList[i]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    if smiles not in selected_mol_set:
      selected_mol_set.add(smiles)
      selected_mol_list.append(smiles)
      if take_mol_subset and len(selected_mol_set) >= mols_to_choose:
        break
      
  if get_all_compounds:
    file_name_list = ['../davis_data/restructured.csv', '../metz_data/restructured_unique.csv',
      '../KIBA_data/restructured_unique.csv', '../full_toxcast/restructured.csv']
    other_mols = get_smiles_from_prev(file_name_list=file_name_list)
    for smiles in other_mols:
      if smiles not in selected_mol_set:
        selected_mol_set.add(smiles)
        selected_mol_list.append(smiles)
  
  loading_functions = {
    'kiba': dcCustom.molnet.load_kiba,
    'toxcast': dcCustom.molnet.load_toxcast,
    'all_kinase': dcCustom.molnet.load_kinases,
    'tc_kinase':dcCustom.molnet.load_tc_kinases,
    'tc_full_kinase': dcCustom.molnet.load_tc_full_kinases
  }  

  tasks, _, _ = loading_functions[dataset_used](featurizer="ECFP", currdir="../")  
  
  prot_list = []

  for path in prot_desc_path_list:
    load_prot_dict(prot_list, path, 1, 2, pairs_to_choose=prot_pairs_to_choose)

  start_writing = time.time()
  suffix = {
    'toxcast': '_tc',
    'kiba': '_kiba'
  }
  fname = output_prefix + suffix[dataset_used] + '.csv'
  with open(fname, 'w', newline='') as csvfile:
    fieldnames = tasks + ['smiles', 'proteinName', 'protein_dataset']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for mol in selected_mol_list:
      for prot in prot_list:
        prot_source_and_name = prot.get_name()
        out_line = {'smiles': mol, 'proteinName': prot_source_and_name[1], 
          'protein_dataset': prot_source_and_name[0]}
        line_values = dict(zip(tasks, [0]*len(tasks)))
        out_line.update(line_values)
        writer.writerow(out_line)

  end_writing = time.time()
  print("Time spent in writing: ", end_writing - start_writing)

def synthesize_ranking(prediction_file, output_file, direction=True, dataset_used='toxcast', 
  weigh_by_occurrence=False, AR_toxcast_codes=[]):
  assert dataset_used in ['toxcast', 'kiba']
  csv_data = {
    'toxcast': '../full_toxcast/restructured.csv',
    'kiba': '../KIBA_data/restructured_unique.csv'
  }
  df = pd.read_csv(csv_data[dataset_used], header=0, index_col=False)
  if dataset_used == 'toxcast':
    tasks, _, _ = dcCustom.molnet.load_toxcast(featurizer="ECFP", currdir="../")
  elif dataset_used == 'kiba':
    tasks, _, _ = dcCustom.molnet.load_kiba(featurizer="ECFP", currdir="../", cross_validation=True, 
      split_warm=True, filter_threshold=6)

  preds_df = pd.read_csv(prediction_file, header=0, index_col=False)
  compounds = preds_df.loc[:, 'Compound']
  prot_names = preds_df.loc[:, 'proteinName']
  prot_sources = preds_df.loc[:, 'protein_dataset']
  composite_preds = np.zeros_like(preds_df.loc[:, tasks[0]])
  if weigh_by_occurrence:
    datapoints = []
    for task in tasks:
      task_vector = np.asarray(df.loc[:, task])
      datapoints.append(np.count_nonzero(~np.isnan(task_vector)))
    datapoints = np.array(datapoints)
    fractions = []
    for i in range(len(datapoints)):
      fractions.append(datapoints[i]/datapoints.sum())    
    for j in range(len(tasks)):
      task = tasks[j]
      pred_task = preds_df.loc[:, task] * fractions[j]
      composite_preds += pred_task
  else:
    AR_toxcast_codes = ['toxcast_' + code for code in AR_toxcast_codes]
    AR_coef_dict = dict(zip(AR_toxcast_codes, AR_antagonist_score_coef))
    for task_code in AR_toxcast_codes:
      pred_task_vector = np.asarray(preds_df.loc[:, task_code])
      composite_preds += pred_task_vector * AR_coef_dict[task_code]   
    
  if direction:
    neg_composite_preds = -1 * composite_preds
    sorted_indices = neg_composite_preds.argsort()
  else:
    sorted_indices = composite_preds.argsort()
  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'proteinName', 'protein_dataset', 'synthesized_score']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in sorted_indices:
      out_line = {'smiles': compounds[i], 'proteinName': prot_names[i], 
        'protein_dataset': prot_sources[i], 'synthesized_score': composite_preds[i]}
      writer.writerow(out_line)  

def compare(file_1, file_2, cutoff=None, exclude_prot=None):
  df_1 = pd.read_csv(file_1, header=0, index_col=False)
  df_2 = pd.read_csv(file_2, header=0, index_col=False)
  if cutoff is not None:
    df_1 = df_1.head(cutoff)
    df_2 = df_2.head(cutoff)
  if exclude_prot is not None:
    correct_type = isinstance(exclude_prot, list)
    assert correct_type

  pred_triplets_set_1 = set()
  pred_triplets_set_2 = set()
  for row in df_1.itertuples():
    if exclude_prot is not None:
      if (row[3], row[2]) in exclude_prot:
        continue
    pred_triplets_set_1.add((row[1], row[2], row[3]))
  for row in df_2.itertuples():
    if exclude_prot is not None:
      if (row[3], row[2]) in exclude_prot:
        continue
    pred_triplets_set_2.add((row[1], row[2], row[3]))
  intersec = pred_triplets_set_1.intersection(pred_triplets_set_2)
  print(len(intersec))
  pdb.set_trace()

def get_invalid_smiles(out_file='invalid_smiles.csv'):
  err_log = open('../logs/error.log', 'r')
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['invalid_smiles']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for line in err_log:
      raw_smiles = re.search(r"'.+'", line).group(0)
      smiles=raw_smiles[1:-1]
      out_line = {'invalid_smiles': smiles}
      writer.writerow(out_line)
  err_log.close()

def get_avg(input_files_list = [], output_file_name = 'avg_ar_tc.csv', exclude_prot=[], direction=True):
  assert isinstance(input_files_list, list)
  assert len(input_files_list) > 0
  record_dict_list = []
  triplet_list = []
  for i, input_file in enumerate(input_files_list):
    df = pd.read_csv(input_file, header=0, index_col=False)
    record_dict = {}
    for row in df.itertuples():
      if len(exclude_prot) > 0:
        if (row[3], row[2]) in exclude_prot:
          continue
      triplet = (row[1], row[2], row[3])
      if triplet not in record_dict:
        record_dict[triplet] = row[4]
        if i == 0:
          triplet_list.append(triplet)
    record_dict_list.append(record_dict)
  avg_val_list = []
  for triplet in triplet_list:
    sum_val = 0
    for record_dict in record_dict_list:
      sum_val += record_dict[triplet]
    avg_val = sum_val/len(record_dict_list)
    avg_val_list.append(avg_val)
  avg_val_arr = np.asarray(avg_val_list)
  if direction:
    sorted_indices = np.argsort(-1 * avg_val_arr)
  else:
    sorted_indices = np.argsort(avg_val_arr)
  
  with open(output_file_name, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'proteinName', 'protein_dataset', 'avg_score']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in sorted_indices:
      triplet = triplet_list[i]
      out_line = {'smiles': triplet[0], 'proteinName': triplet[1], 
        'protein_dataset': triplet[2], 'avg_score': avg_val_list[i]}
      writer.writerow(out_line)       
  
  
def calculate_mean_activity(pred_file, top_n_list = [100, 1000, 54000], exclude_prot=[],
  out_file='mean_logGI50.csv'):
  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  top_n_list = [len(df_avg)] + top_n_list
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  panel = 'Prostate'
  dict_list = []
  for top_n in top_n_list:
    df_avg_subset = df_avg.head(top_n)
    compounds = df_avg_subset.loc[:, 'smiles']
    compounds_set = set(compounds)        
    cell_lines_to_values = {}  
    
    for row in df_nci60.itertuples():    
      if not re.search(panel, row[1], re.I):
        continue
      if np.isnan(row[7]):
        continue            
      if row[3] not in compounds_set:
        continue
      if row[2] not in cell_lines_to_values:
        cell_lines_to_values[row[2]] = []
      cell_lines_to_values[row[2]].append(row[7])
    dict_list.append(cell_lines_to_values)

  cell_line_list = list(dict_list[0])
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'mean value', 
      'standard deviation']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel}
    for cell_line in cell_line_list:
      out_line.update({'cell line': cell_line})
      for i, top_n in enumerate(top_n_list):
        cell_lines_to_values = dict_list[i]
        if cell_line not in cell_lines_to_values:
          continue
        values = cell_lines_to_values[cell_line]
        values = np.asarray(values)
        out_line.update({'top_n': top_n, 'num_observation': len(values), 'mean value': np.mean(values),
          'standard deviation': np.std(values)})
        writer.writerow(out_line) 

if __name__ == "__main__":
  #dataset = 'toxcast'
  dataset = 'kiba'
  # produce_dataset(dataset_used=dataset, prot_desc_path_list=['../full_toxcast/prot_desc.csv'], 
  #   get_all_compounds=True, take_mol_subset=False, prot_pairs_to_choose=AR_list_s + ER_list_s)
  # synthesize_ranking('preds_arer_kiba_graphconv.csv', 'ordered_arer_kiba_gc.csv', weigh_by_occurrence=True,
  #   AR_toxcast_codes=AR_toxcast_codes, dataset_used=dataset, direction=False)   
  # synthesize_ranking('preds_tc_graphconv.csv', 'synthesized_values_gc.csv', 
  #   direction=True, dataset_used=dataset)
  #compare('ordered_arer_kiba_ecfp.csv', 'ordered_arer_tc_ecfp.csv', cutoff=2000, exclude_prot=ER_list_s)
  #get_invalid_smiles(out_file = 'invalid_smiles.csv')  
  #get_avg(input_files_list=['ordered_arer_tc_ecfp.csv', 'ordered_arer_tc_gc.csv'], exclude_prot=ER_list_s)
  calculate_mean_activity('avg_ar_tc.csv')