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

def get_canonical_smiles_dict(file_name='invalid_smiles_canonicalized.csv', reverse=False):
  smiles_file = pd.read_csv(file_name, header=0, index_col=False)
  invalid_smiles = smiles_file.iloc[:, 0]
  canonical_smiles = smiles_file.iloc[:, 1]
  if reverse:
    smiles_dict = dict(zip(canonical_smiles, invalid_smiles))    
  else:
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
  
  
def calculate_mean_activity(pred_file, top_n_list = [15000, 1000, 100], exclude_prot=[],
  out_file='mean_logGI50.csv', threshold=2000):
  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  panel = 'Prostate'
  cell_lines_to_pairs = {}
  cline_to_topn_list = {}
  cline_and_topn_to_value_list = {}
  compounds = df_avg.loc[:,'smiles']
  compounds_set = set(compounds)
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  
  for row in df_nci60.itertuples():    
    if not re.search(panel, row[1], re.I):
      continue
    if np.isnan(row[7]):
      continue     
    smiles = row[3]    
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    # if smiles not in compounds_set:
    #   pdb.set_trace()
    #   #continue
    if row[2] not in cell_lines_to_pairs:
      cell_lines_to_pairs[row[2]] = {}
    cell_lines_to_pairs[row[2]].update({smiles: row[7]})

  for cline in cell_lines_to_pairs:
    compound_to_value = cell_lines_to_pairs[cline]
    size = len(compound_to_value)
    if size < threshold:
      continue
    cline_top_n_list = [size] + top_n_list    
    cline_to_topn_list[cline] = cline_top_n_list
    
    for i, top_n in enumerate(cline_top_n_list):
      if top_n > size:
        continue
      if i < len(cline_top_n_list) - 1:
        # Sanity check.
        assert cline_top_n_list[i + 1] <= top_n
      compound_to_value_subset = {} 
      for row in df_avg.itertuples():
        smiles = row[1]
        if smiles in compound_to_value:
          compound_to_value_subset[smiles] = compound_to_value[smiles]        
          if len(compound_to_value_subset) >= top_n:
            #pdb.set_trace()
            compound_to_value = compound_to_value_subset
            cline_and_topn_to_value_list[(cline, top_n)] = list(compound_to_value.values())
            break

  cell_line_list = list(cell_lines_to_pairs)
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'mean value', 
      'standard deviation']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel}
    for cell_line in cell_line_list:
      out_line.update({'cell line': cell_line})
      if cell_line not in cline_to_topn_list:
        continue
      this_top_n_list = cline_to_topn_list[cell_line]
      for top_n in this_top_n_list:
        pair = (cell_line, top_n)
        if pair not in cline_and_topn_to_value_list:
          continue
        values = cline_and_topn_to_value_list[pair]
        values = np.asarray(values)
        out_line.update({'top_n': top_n, 'num_observation': len(values), 'mean value': np.mean(values),
          'standard deviation': np.std(values)})
        writer.writerow(out_line)

def get_dcg(value_array):
  length = len(value_array)
  terms = np.empty_like(value_array, dtype=np.float64)
  for i in range(length):
    terms[i] = value_array[i]/np.log2(i + 2)
  return np.sum(terms)

def ndcg_tool(ordered_cpd_list, panel_cline_and_compound_to_value, sorted_values_array, 
  cell_line=None, panel='Prostate', direction=True):
  # If direction is True, the smaller values are expected to rank higher.
  values_of_predicted_ranking = []
  for cpd in ordered_cpd_list:
    values_of_predicted_ranking.append(panel_cline_and_compound_to_value[(panel, cell_line, cpd)])
  values_of_predicted_ranking = np.asarray(values_of_predicted_ranking)
  total_num = len(ordered_cpd_list)
  if direction:
    max_val = sorted_values_array[len(sorted_values_array) - 1]
  else:
    min_val = sorted_values_array[len(sorted_values_array) - 1]
  sorted_values_array = sorted_values_array[:total_num]
  if direction:
    sorted_values_array = max_val - sorted_values_array
    values_of_predicted_ranking = max_val - values_of_predicted_ranking
  else:
    sorted_values_array = sorted_values_array - min_val
    values_of_predicted_ranking = values_of_predicted_ranking - min_val
  dcg = get_dcg(values_of_predicted_ranking)
  idcg = get_dcg(sorted_values_array)
  return dcg/idcg, dcg, idcg

def calculate_ndcg(pred_file, top_n_list = [12000, 1000, 100], exclude_prot=[],
  out_file='normalized_dcg_logGI50.csv', threshold=2000):
  # Calculates the normalized discounted cumulative gain using the logGI50 value as the relevance score.
  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  dict_list = []
  panel = 'Prostate'
  panel_cline_and_compound_to_value = {}
  cell_line_list = []
  compounds = df_avg.loc[:, 'smiles']
  compounds_set = set(compounds)
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  # TODO: unfinished.
  for row in df_nci60.itertuples():     
    if not re.search(panel, row[1], re.I):
      continue
    if np.isnan(row[7]):
      continue  
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]  
    if row[2] not in set(cell_line_list):
      cell_line_list.append(row[2])
    triplet = (panel, row[2], smiles)
    panel_cline_and_compound_to_value[triplet] = row[7]  
  
  values_list = [v for v in panel_cline_and_compound_to_value.values()]
  values_array = np.asarray(values_list)
  sorted_values_array = np.sort(values_array)

  for cline in cell_line_list:
    for top_n in top_n_list:
      
      assert len(compounds) == len(compounds_set)        
      cell_lines_to_ordered_compounds = {}
      cell_lines_to_compound_set = {}
      
      for row in df_nci60.itertuples():    
        if not re.search(panel, row[1], re.I):
          continue
        if np.isnan(row[7]):
          continue            
        smiles = row[3]
        if smiles in invalid_to_canon_smiles:
          smiles = invalid_to_canon_smiles[smiles]
        if row[2] not in cell_lines_to_compound_set:
          cell_lines_to_compound_set[row[2]] = set()      
        cell_lines_to_compound_set[row[2]].add(smiles)

      for key in cell_lines_to_compound_set:
        ordered_cpd_list = [cpd for cpd in compounds if cpd in cell_lines_to_compound_set[key] ]
        cell_lines_to_ordered_compounds[key] = ordered_cpd_list  
      dict_list.append(cell_lines_to_ordered_compounds)  

  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'nDCG', 'DCG', 'iDCG']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel}
    for cell_line in cell_line_list:
      out_line.update({'cell line': cell_line})
      for i, top_n in enumerate(top_n_list):
        cell_lines_to_ordered_compounds = dict_list[i]
        if cell_line not in cell_lines_to_ordered_compounds:
          continue
        ordered_cpd_list = cell_lines_to_ordered_compounds[cell_line]
        normalized_dcg, dcg, idcg = ndcg_tool(ordered_cpd_list, panel_cline_and_compound_to_value, 
          sorted_values_array, cell_line=cell_line, panel=panel)
        out_line.update({'top_n': top_n, 'num_observation': len(ordered_cpd_list), 
          'nDCG': normalized_dcg, 'DCG': dcg, 'iDCG': idcg})
        writer.writerow(out_line)

def plot_values(panel='Prostate', clines=['DU-145', 'PC-3'], plot_all_panels=True, threshold=100):  
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  panel_list = []
  panel_set = set()
  if plot_all_panels:
    panel_clines_to_value_list = {}
    clines = []
  else:
    panel_list = [panel] * len(clines)
    tuples_list = list(zip(panel_list, clines))
    panel_clines_to_value_list = dict(zip(tuples_list, [[], []]))
  for row in df_nci60.itertuples():
    if plot_all_panels:
      panel = row[1].rstrip()
      if panel not in panel_set:
        panel_list.append(panel)
        panel_set.add(panel)
    else:
      if not re.search(panel, row[1], re.I):
        continue

    if np.isnan(row[7]):
      continue

    if plot_all_panels:
      cline = row[2].rstrip()
      if (panel, cline) not in panel_clines_to_value_list:
        panel_clines_to_value_list[(panel, cline)] = []
      panel_clines_to_value_list[(panel, cline)].append(row[7])
    else:
      for cline in clines:
        if re.search(cline, row[2], re.I):
          panel_clines_to_value_list[(panel, cline)].append(row[7])

  for key in panel_clines_to_value_list: 
    values = np.asarray(panel_clines_to_value_list[key])
    if len(values) <= threshold:
      continue
    num_bins = 50
    fig, ax = plt.subplots()  
    #n, bins, patches = ax.hist(interactions, num_bins, density=1)
    min_val = values.min    
    max_val = values.max
    #ax.hist(values, num_bins, range=(np.floor(min_val), np.ceil(max_val)))
    ax.hist(values, num_bins)
    ax.set_xlabel('logGI50 values')
    ax.set_ylabel('Occurrence')
    ax.set_title('Histogram of logGI50 values for cell line ' + key[1]) 
    cell_line = key[1]
    cell_line = cell_line.replace('/', '.')   
    image_name = "plots/" + key[0] + '_' + cell_line + ".png"
    plt.savefig(image_name)
    plt.close()

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
  #calculate_ndcg('avg_ar_tc.csv')
  #plot_values()
  