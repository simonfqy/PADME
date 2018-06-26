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
ER_list = [('toxcast', 'P03372'), ('toxcast', 'P19785'), ('toxcast', 'P49884'), 
  ('toxcast', 'Q92731')]

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
  return smiles_list
    
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
  output_prefix='AR_ER_intxn'):
  
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
  weigh_by_occurrence=False):
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

  if weigh_by_occurrence:
    datapoints = []
    for task in tasks:
      task_vector = np.asarray(df.loc[:, task])
      datapoints.append(np.count_nonzero(~np.isnan(task_vector)))
    datapoints = np.array(datapoints)
    fractions = []
    for i in range(len(datapoints)):
      fractions.append(datapoints[i]/datapoints.sum())

    preds_df = pd.read_csv(prediction_file, header=0, index_col=False)
    compounds = preds_df.loc[:, 'Compound']
    prot_names = preds_df.loc[:, 'proteinName']
    prot_sources = preds_df.loc[:, 'protein_dataset']
    composite_preds = np.zeros_like(preds_df.loc[:, tasks[0]])
    for j in range(len(tasks)):
      task = tasks[j]
      pred_task = preds_df.loc[:, task] * fractions[j]
      composite_preds += pred_task
  else:
    # TODO: finish it.
    pass
    
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

def compare(file_1, file_2, cutoff=None):
  df_1 = pd.read_csv(file_1, header=0, index_col=False)
  df_2 = pd.read_csv(file_2, header=0, index_col=False)
  if cutoff is not None:
    df_1 = df_1.head(cutoff)
    df_2 = df_2.head(cutoff)
  pred_triplets_set_1 = set()
  pred_triplets_set_2 = set()
  for row in df_1.itertuples():
    pred_triplets_set_1.add((row[1], row[2], row[3]))
  for row in df_2.itertuples():
    pred_triplets_set_2.add((row[1], row[2], row[3]))
  intersec = pred_triplets_set_1.intersection(pred_triplets_set_2)
  print(len(intersec))
  #pdb.set_trace()

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

if __name__ == "__main__":
  dataset = 'toxcast'
  #dataset = 'kiba'
  produce_dataset(dataset_used=dataset, prot_desc_path_list=['../full_toxcast/prot_desc.csv'], 
    get_all_compounds=True, take_mol_subset=False, prot_pairs_to_choose=AR_list+ER_list)
  #synthesize_ranking('preds_tc_graphconv.csv', 'ordered_gc.csv',   
  # synthesize_ranking('preds_tc_graphconv.csv', 'synthesized_values_gc.csv', 
  #   direction=True, dataset_used=dataset)
  #compare('ordered_kiba.csv', 'synthesized_values.csv', cutoff=2000)
  #get_invalid_smiles(out_file = 'invalid_smiles.csv')
  