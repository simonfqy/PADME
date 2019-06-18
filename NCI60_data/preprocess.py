from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
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
import scipy
from dcCustom.feat import Protein
from dcCustom.metrics import concordance_index
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols


AR_list = [('toxcast', 'P10275'), ('toxcast', 'O97775'), ('toxcast', 'P15207')]
AR_list_s = [('toxcast', 'P10275')]
ER_list = [('toxcast', 'P03372'), ('toxcast', 'P19785'), ('toxcast', 'P49884'), 
  ('toxcast', 'Q92731')]
ER_list_s = [('toxcast', 'P03372')]
# AR_toxcast_codes = ['6620', '8110', '8010', '7100', '7200', '3100', '7300', '8100', '8000']
# AR_antagonist_score_coef = [1, 0.5, 0.5, -0.5, -0.5, -1, -1/3, -1/3, -1/3]
AR_toxcast_codes = ['6620']
AR_antagonist_score_coef = [1]
# AR_toxcast_codes = ['6620', '8110', '8010']
# AR_antagonist_score_coef = [1, 0.5, 0.5]
#AR_antagonist_score_coef = [1.] + [0.]*8

# AR_toxcast_codes = ['6620', '7100', '7200', '3100', '7300', '8100', '8000']
# AR_agonist_score_coef = [1, 0.5, 0.5, 1, 1/3, 1/3, 1/3]

PROT_desc_path_list = ['../davis_data/prot_desc.csv', '../metz_data/prot_desc.csv', 
    '../KIBA_data/prot_desc.csv', '../full_toxcast/prot_desc.csv']

def get_toxcast_code_dict(file_name='../full_toxcast/Assay_UniprotID_subgroup.csv'):
  tc_code_and_protID_to_assays = {}
  grouping_file = pd.read_csv(file_name, header=0, index_col=False)
  for row in grouping_file.itertuples():
    code = row[3]
    tc_code = 'toxcast_' + str(code)
    uniprot_ID = row[2]
    pair = (tc_code, uniprot_ID) 
    if pair not in tc_code_and_protID_to_assays:
      tc_code_and_protID_to_assays[pair] = set()
    tc_code_and_protID_to_assays[pair].add(row[1])
  return tc_code_and_protID_to_assays

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
      
def select_mol_list(dataframe, selected_mol_set, selected_mol_list, 
  invalid_to_canon_smiles={}, take_mol_subset=True, mols_to_choose=2000, 
  panel="Prostate", group_by_celllines=True, cellline_threshold=500, mols_per_cline=500):
  
  if not group_by_celllines:
    GIarray = np.asarray(dataframe.iloc[:, 5])
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
  else:
    cell_line_to_compounds_list = {}
    cell_line_to_values_list = {}
    cell_line_to_invalid_smiles = {}
    for row in dataframe.itertuples():      
      if not re.search(panel, row[1], re.I):
        continue
      if np.isnan(row[6]):
        continue      
      cell_line = row[2]   
      smiles = row[0]
      if smiles in invalid_to_canon_smiles:
        smiles = invalid_to_canon_smiles[smiles]
      if cell_line not in cell_line_to_compounds_list:
        cell_line_to_compounds_list[cell_line] = []
        cell_line_to_values_list[cell_line] = []
        cell_line_to_invalid_smiles[cell_line] = set()
      if smiles in set(cell_line_to_compounds_list[cell_line]):
        cell_line_to_invalid_smiles[cell_line].add(smiles)        
      cell_line_to_compounds_list[cell_line].append(smiles)
      cell_line_to_values_list[cell_line].append(row[6])
    
    for cell_line in cell_line_to_compounds_list:
      compound_list = cell_line_to_compounds_list[cell_line]      
      values_list = cell_line_to_values_list[cell_line]
      smiles_to_exclude = cell_line_to_invalid_smiles[cell_line]
      values_list = [val for (cpd, val) in zip(compound_list, values_list
        ) if cpd not in smiles_to_exclude]
      compound_list = [cpd for cpd in compound_list if cpd not in smiles_to_exclude]
      pdb.set_trace()
      if len(values_list) < cellline_threshold:
        continue
      value_array = np.asarray(values_list)
      sorted_indices = np.argsort(value_array)
      for i in range(len(sorted_indices)):
        if i >= mols_per_cline:
          continue
        smiles = compound_list[sorted_indices[i]]
        if smiles not in selected_mol_set:
          selected_mol_set.add(smiles) 
          selected_mol_list.append(smiles)     
      
    
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
  
  select_mol_list(df, selected_mol_set, selected_mol_list, 
    invalid_to_canon_smiles=invalid_to_canon_smiles, mols_to_choose=mols_to_choose)
      
  if get_all_compounds:
    file_name_list = ['../davis_data/restructured.csv', '../metz_data/restructured_unique.csv',
      '../KIBA_data/restructured_unique.csv', '../full_toxcast/restructured.csv']
    other_mols = get_smiles_from_prev(file_name_list=file_name_list)
    for smiles in other_mols:
      if smiles not in selected_mol_set:
        selected_mol_set.add(smiles)
        selected_mol_list.append(smiles)
  assert len(selected_mol_set) == len(selected_mol_list)
  print("len(selected_mol_list): ", len(selected_mol_list))
  
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
  weigh_by_occurrence=False, AR_score_coef = [], AR_toxcast_codes=[]):
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
    AR_coef_dict = dict(zip(AR_toxcast_codes, AR_score_coef))
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

def get_avg(input_files_list = [], output_file_name = 'avg_ar_tc.csv', exclude_prot=[], 
  direction=True):
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
  

def get_highest_similarity_for_mol(fp, fp_list_to_compare):
  max_sim = 0
  for fp_comp in fp_list_to_compare:
    sim = DataStructs.FingerprintSimilarity(fp, fp_comp)
    if sim == 1.0:
      return sim
    if sim > max_sim:
      max_sim = sim
  return max_sim


def get_highest_similarity(input_file, output_file, comparison_file='../full_toxcast/restructured.csv',
  top_compounds_only=True, num_compounds=1500):
  df_avg = pd.read_csv(input_file, header=0, index_col=False)
  if top_compounds_only:
    df_avg = df_avg.head(num_compounds)
  smiles_list = df_avg['smiles']
  avg_scores = df_avg['avg_score']
  # default_mol = Chem.MolFromSmiles('CCCC')
  # default_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(default_mol, 2, nBits=1024)
  # mol2 = Chem.MolFromSmiles('CCCC')
  # fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
  # sim = DataStructs.FingerprintSimilarity(default_fp, fp2)
  df_comparison = pd.read_csv(comparison_file, header=0, index_col=False)
  #df_comparison = df_comparison.head(100)
  comparison_smiles_list = df_comparison['smiles']
  comparison_fp_list = []
  similarity_list = []
  for c_smiles in comparison_smiles_list:
    comp_mol = Chem.MolFromSmiles(c_smiles)
    #comp_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(comp_mol, 2, nBits=1024)
    comp_fp = FingerprintMols.FingerprintMol(comp_mol)
    comparison_fp_list.append(comp_fp)

  for i, smiles in enumerate(smiles_list):
    mol_to_test = Chem.MolFromSmiles(smiles)
    #fp_to_test = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_to_test, 2, nBits=1024)
    fp_to_test = FingerprintMols.FingerprintMol(mol_to_test)
    similarity_list.append(get_highest_similarity_for_mol(fp_to_test, comparison_fp_list))
    if i%500 == 0:
      print(i)

  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'avg_score', 'max_similarity']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i, smiles in enumerate(smiles_list):
      out_line = {'smiles': smiles, 'avg_score': avg_scores[i], 
        'max_similarity': similarity_list[i]}
      writer.writerow(out_line)  
 

def filter_similarity(input_file, output_file, image_folder='./mol_images', threshold=0.85, 
  draw_2d_image=True, limit=False, limit_lines=120):
  # Both input_file and output_file should be csv files.
  df_sim = pd.read_csv(input_file, header=0, index_col=False)
  smiles_array = df_sim['smiles']
  scores_array = df_sim['avg_score']
  similarity_array = df_sim['max_similarity']
  selected_indices_list = []
  for i, sim in enumerate(similarity_array):
    if sim >= threshold:
      continue
    selected_indices_list.append(i)
    if draw_2d_image:
      source_line_num = i + 2
      target_line_num = len(selected_indices_list) + 1
      mol = Chem.MolFromSmiles(smiles_array[i])
      Draw.MolToFile(mol, 
        image_folder + '/' + str(source_line_num) + '_' + str(target_line_num) + '.png',
        size=(700,700))

  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'avg_score', 'max_similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, ind in enumerate(selected_indices_list):
      if limit and i + 1 > limit_lines:
        break
      out_line = {'smiles': smiles_array[ind], 'avg_score': scores_array[ind], 
        'max_similarity': similarity_array[ind]}
      writer.writerow(out_line)

  
def calculate_mean_activity(pred_file, top_n_list = [15000, 1000, 100], exclude_prot=[],
  out_file='mean_logGI50_breast.csv', threshold=6000):

  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  panel = 'Breast'
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

def calculate_ndcg(pred_file, top_n_list = [15000, 1000, 500, 100, 50, 10], exclude_prot=[],
  out_file='normalized_dcg_logGI50.csv', threshold=2000):
  # Calculates the normalized discounted cumulative gain using the logGI50 value as the relevance score.
  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)  
  panel = 'Prostate'
  cline_and_topn_to_ordered_compounds = {}
  panel_cline_and_compound_to_value = {}
  cell_line_list = []
  cline_to_topn_list = {}
  compounds = df_avg.loc[:, 'smiles']
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
    if row[2] not in set(cell_line_list):
      cell_line_list.append(row[2])
    triplet = (panel, row[2], smiles)
    panel_cline_and_compound_to_value[triplet] = row[7]  
  
  values_list = [v for v in panel_cline_and_compound_to_value.values()]
  values_array = np.asarray(values_list)
  sorted_values_array = np.sort(values_array)

  for cline in cell_line_list:
    compound_to_value = {}
    for triplet in panel_cline_and_compound_to_value:
      if triplet[1] == cline:
        # Make sure that no duplicate smiles.
        assert triplet[2] not in compound_to_value
        compound_to_value[triplet[2]] = panel_cline_and_compound_to_value[triplet]
    size = len(compound_to_value)
    if size < threshold:
      continue
    cline_top_n_list = [size] + top_n_list
    cline_to_topn_list[cline] = cline_top_n_list

    for i, top_n in enumerate(cline_top_n_list):
      if top_n > size:
        continue
      if i < len(cline_top_n_list) - 1:
        assert cline_top_n_list[i + 1] <= top_n         
      pair = (cline, top_n)   
      compound_list = []
      compound_to_value_subset = {}   

      for row_pred in df_avg.itertuples():
        smiles = row_pred[1]
        if smiles in compound_to_value:
          compound_to_value_subset[smiles] = compound_to_value[smiles]
          compound_list.append(smiles)
          if len(compound_to_value_subset) >= top_n:
            compound_to_value = compound_to_value_subset
            break
      cline_and_topn_to_ordered_compounds[pair] = compound_list      
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'nDCG', 'DCG', 'iDCG']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel}
    for cell_line in cell_line_list:
      out_line.update({'cell line': cell_line})
      if cell_line not in cline_to_topn_list:
        continue
      cline_top_n_list = cline_to_topn_list[cell_line]
      for top_n in cline_top_n_list:
        pair = (cell_line, top_n)
        if pair not in cline_and_topn_to_ordered_compounds:
          continue
        ordered_cpd_list = cline_and_topn_to_ordered_compounds[pair]
        normalized_dcg, dcg, idcg = ndcg_tool(ordered_cpd_list, panel_cline_and_compound_to_value, 
          sorted_values_array, cell_line=cell_line, panel=panel)
        out_line.update({'top_n': top_n, 'num_observation': len(ordered_cpd_list), 
          'nDCG': normalized_dcg, 'DCG': dcg, 'iDCG': idcg})
        writer.writerow(out_line)

def calc_subsets_cindex(pred_file, top_n_list = [15000, 1000, 500, 100, 50, 10], exclude_prot=[],
  out_file='subset_cindex_logGI50.csv', threshold=1500):
  # Calculate the cindex on the ranking of AR score using logGI50 as ground truth.
  df_avg = pd.read_csv(pred_file, header=0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)  
  panel = 'Prostate'
  cline_and_topn_to_ordered_compounds = {}
  panel_cline_and_compound_to_value = {}
  cell_line_list = []
  cline_to_topn_list = {}
  smiles_to_ar_score = {}
  invalid_to_canon_smiles = get_canonical_smiles_dict()

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
    panel_cline_and_compound_to_value[triplet] = -1 * row[7]  
  
  for row_pred in df_avg.itertuples():
    smiles = row_pred[1]
    avg_score = row_pred[4]
    assert smiles not in smiles_to_ar_score
    smiles_to_ar_score[smiles] = avg_score

  for cline in cell_line_list:
    compound_to_value = {}
    for triplet in panel_cline_and_compound_to_value:
      if triplet[1] == cline:
        # Make sure that no duplicate smiles.
        assert triplet[2] not in compound_to_value
        compound_to_value[triplet[2]] = panel_cline_and_compound_to_value[triplet]
    size = len(compound_to_value)
    if size < threshold:
      continue
    cline_top_n_list = [size] + top_n_list
    cline_to_topn_list[cline] = cline_top_n_list

    for i, top_n in enumerate(cline_top_n_list):
      if top_n > size:
        continue
      if i < len(cline_top_n_list) - 1:
        assert cline_top_n_list[i + 1] <= top_n         
      pair = (cline, top_n)   
      compound_list = []
      compound_to_value_subset = {}   

      for row_pred in df_avg.itertuples():
        smiles = row_pred[1]
        if smiles in compound_to_value:
          compound_to_value_subset[smiles] = compound_to_value[smiles]
          compound_list.append(smiles)
          if len(compound_to_value_subset) >= top_n:
            compound_to_value = compound_to_value_subset
            break
      cline_and_topn_to_ordered_compounds[pair] = compound_list      
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'cindex']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel}
    for cell_line in cell_line_list:
      out_line.update({'cell line': cell_line})
      if cell_line not in cline_to_topn_list:
        continue
      cline_top_n_list = cline_to_topn_list[cell_line]
      for top_n in cline_top_n_list:
        pair = (cell_line, top_n)
        if pair not in cline_and_topn_to_ordered_compounds:
          continue
        ordered_cpd_list = cline_and_topn_to_ordered_compounds[pair]
        
        gi_list = [panel_cline_and_compound_to_value[(panel, cell_line, smiles)] for smiles in 
          ordered_cpd_list]
        gi_array = np.asarray(gi_list)
        ar_list = [smiles_to_ar_score[smiles] for smiles in ordered_cpd_list]
        ar_array = np.asarray(ar_list)
        cindex = concordance_index(gi_array, ar_array)
        out_line.update({'top_n': top_n, 'num_observation': len(ordered_cpd_list), 
          'cindex': cindex})
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
    min_val = values.min    
    max_val = values.max
    ax.hist(values, num_bins)
    ax.set_xlabel('logGI50 values')
    ax.set_ylabel('Occurrence')
    ax.set_title('Histogram of logGI50 values for cell line ' + key[1]) 
    cell_line = key[1]
    cell_line = cell_line.replace('/', '.')   
    image_name = "plots/" + key[0] + '_' + cell_line + ".png"
    plt.savefig(image_name)
    plt.close()
    
def get_spearman_tuple(base_panel_to_clines, panels_to_clines, panel_and_cline_to_smiles, 
  panel_cline_and_smiles_to_value, smiles_to_ar_score, base_panel='Prostate', 
  threshold = 0, smiles_to_exclude=set()):

  cell_line_pair_to_spearman_tuple = {}
  for cell_line in base_panel_to_clines[base_panel]:
    pair = (base_panel, cell_line)
    base_smiles_list = panel_and_cline_to_smiles[pair]
    base_smiles_set = set(base_smiles_list)
    if len(base_smiles_list) < threshold:
      continue
    assert len(base_smiles_set & smiles_to_exclude) == 0
    base_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(base_panel, cell_line, 
      smiles)] for smiles in base_smiles_list])
    base_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in base_smiles_list])
    rho, pval = scipy.stats.spearmanr(base_nci60_values, base_ar_values)
    cell_line_pair = (cell_line, cell_line)
    assert cell_line_pair not in cell_line_pair_to_spearman_tuple
    cell_line_pair_to_spearman_tuple[cell_line_pair] = (rho, pval, len(base_nci60_values))
    for compare_panel in panels_to_clines:
      if base_panel == compare_panel:
        continue
      for compare_cline in panels_to_clines[compare_panel]:     
        this_pair = (compare_panel, compare_cline)
        compare_smiles_list = panel_and_cline_to_smiles[this_pair]
        compare_smiles_set = set(compare_smiles_list)
        assert len(compare_smiles_set & smiles_to_exclude) == 0
        intersecting_smiles_list = [smiles for smiles in base_smiles_list if smiles in 
          compare_smiles_set]
        if len(intersecting_smiles_list) < threshold:
          continue  

        cell_line_pair = (compare_cline, compare_cline) 
        if cell_line_pair not in cell_line_pair_to_spearman_tuple:          
          compare_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(compare_panel,
            compare_cline, smiles)] for smiles in compare_smiles_list])
          compare_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in compare_smiles_list])
          rho, pval = scipy.stats.spearmanr(compare_nci60_values, compare_ar_values)       
          cell_line_pair_to_spearman_tuple[cell_line_pair] = (rho, pval, len(compare_nci60_values))
        cell_line_pair = (cell_line, compare_cline)
        intersecting_base_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(base_panel,
          cell_line, smiles)] for smiles in intersecting_smiles_list])
        intersecting_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in 
          intersecting_smiles_list])
        rho, pval = scipy.stats.spearmanr(intersecting_base_nci60_values, intersecting_ar_values)
        cell_line_pair_to_spearman_tuple[cell_line_pair] = (rho, pval, 
          len(intersecting_base_nci60_values))
        cell_line_pair = (compare_cline, cell_line)
        intersecting_compare_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(compare_panel,
          compare_cline, smiles)] for smiles in intersecting_smiles_list])
        rho, pval = scipy.stats.spearmanr(intersecting_compare_nci60_values, 
          intersecting_ar_values)
        cell_line_pair_to_spearman_tuple[cell_line_pair] = (rho, pval, 
          len(intersecting_compare_nci60_values))
  return cell_line_pair_to_spearman_tuple

def get_selective_compounds(base_cell_line='DU-145', out_file='ordered_compounds.csv'):
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  panel_and_cline_to_smiles = {}
  base_cline_compound_list = []
  base_cline_smiles_to_value = {}
  smiles_to_activity_list = {}
  smiles_to_exclude = set()
  for row in df_nci60.itertuples():
    if np.isnan(row[7]):
      continue
    this_panel = row[1].rstrip()
    cell_line = row[2].rstrip()      
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    pair = (this_panel, cell_line)
    if pair not in panel_and_cline_to_smiles:
      panel_and_cline_to_smiles[pair] = set()
    if smiles in panel_and_cline_to_smiles[pair]:
      smiles_to_exclude.add(smiles)
    panel_and_cline_to_smiles[pair].add(smiles)
    
    active = row[4]
    if cell_line == base_cell_line and active == 1:
      base_cline_compound_list.append(smiles)
      base_cline_smiles_to_value[smiles] = row[7]
    if smiles not in smiles_to_activity_list:
      smiles_to_activity_list[smiles] = []
    if cell_line != base_cell_line:
      smiles_to_activity_list[smiles].append(active)
      
  base_cline_compound_list = [smiles for smiles in base_cline_compound_list if smiles not 
    in smiles_to_exclude]  
  score_list = []
  compound_to_activity_ratio_triplet = {}
  for smiles in base_cline_compound_list:
    activity_list = smiles_to_activity_list[smiles]
    ratio = sum(activity_list)/len(activity_list)
    score = ratio + base_cline_smiles_to_value[smiles]/8
    score_list.append(score)
    assert smiles not in compound_to_activity_ratio_triplet
    compound_to_activity_ratio_triplet[smiles] = (sum(activity_list), len(activity_list), ratio)
    
  score_array = np.asarray(score_list)
  sorted_indices = np.argsort(score_array)
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'DU-145 logGI50', 'active_ratio', 'total active', 'total occurrence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  
    for ind in sorted_indices:
      smiles = base_cline_compound_list[ind]
      logGI50 = base_cline_smiles_to_value[smiles]
      triplet = compound_to_activity_ratio_triplet[smiles]
      out_line = {'smiles': smiles, 'DU-145 logGI50': logGI50, 'active_ratio': triplet[2],
        'total active': triplet[0], 'total occurrence': triplet[1]}
      writer.writerow(out_line)
  
def calc_spearmanr(pred_file, base_cell_lines=['DU-145', 'PC-3'], panels_for_comparison=['Breast'],
  base_panel='Prostate', out_file='ar_logGI50_spearmanr.csv', threshold=1500, select_subset=False,
  num_compounds=250, compound_file='ordered_compounds.csv'):
  
  df_avg = pd.read_csv(pred_file, header = 0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  if select_subset:
    df_top_compounds = pd.read_csv(compound_file, header=0, index_col=False)
    df_top_compounds = df_top_compounds.head(num_compounds)
    smiles_subset = set(df_top_compounds.loc[:, 'smiles'])
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  use_all_panels = False

  panels_to_clines = dict(zip(panels_for_comparison, [set()]*len(panels_for_comparison)))
  base_panel_to_clines = {base_panel: set()}
  panel_and_cline_to_smiles = {}
  panel_cline_and_smiles_to_value = {}
  smiles_to_ar_score = {}
  smiles_to_exclude = set()

  if len(panels_for_comparison) == 0:
    use_all_panels = True
  for row_pred in df_avg.itertuples():
    smiles = row_pred[1]
    avg_score = row_pred[4]
    assert smiles not in smiles_to_ar_score
    smiles_to_ar_score[smiles] = avg_score

  time1 = time.time()
  for row in df_nci60.itertuples():
    if np.isnan(row[7]):
      continue
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    if select_subset and smiles not in smiles_subset:
      continue
    this_panel = row[1].rstrip()
    cell_line = row[2].rstrip()
    
    if this_panel == base_panel:
      if cell_line not in base_panel_to_clines[base_panel]:
        base_panel_to_clines[base_panel].add(cell_line)
    elif use_all_panels:
      if this_panel not in panels_to_clines:
        panels_to_clines[this_panel] = set() 
      if cell_line not in panels_to_clines[this_panel]:
        panels_to_clines[this_panel].add(cell_line)
    else:
      if this_panel in panels_to_clines:
        if cell_line not in panels_to_clines[this_panel]:
          panels_to_clines[this_panel].add(cell_line)
      else:
        continue
    
    pair = (this_panel, cell_line)
    if pair not in panel_and_cline_to_smiles:
      panel_and_cline_to_smiles[pair] = []
    if smiles in set(panel_and_cline_to_smiles[pair]):
      smiles_to_exclude.add(smiles)
    panel_and_cline_to_smiles[pair].append(smiles)
    
    triplet = (this_panel, cell_line, smiles)
    panel_cline_and_smiles_to_value[triplet] = -1 * row[7]

  time2 = time.time()
  print('len(smiles_to_exclude): ', len(smiles_to_exclude))
  print('time used for iterating through NCI60 data: ', time2 - time1)
  for key in panel_and_cline_to_smiles:
    panel_and_cline_to_smiles[key] = [smiles for smiles in panel_and_cline_to_smiles[key]
      if smiles not in smiles_to_exclude]
  
  cell_line_pair_to_spearman_tuple = get_spearman_tuple(base_panel_to_clines, panels_to_clines, 
    panel_and_cline_to_smiles, panel_cline_and_smiles_to_value, smiles_to_ar_score, 
    base_panel=base_panel, threshold = threshold, smiles_to_exclude=smiles_to_exclude)

  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'cell line', 'num_observation', "Spearman's rho", 'p-value',
      'num_intersection', 'rho-intersection', 'pvalue-intersection', 'rho-inter-base-cline',
      'pvalue-inter-base-cline', 'base cell line']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    
    for cell_line in base_panel_to_clines[base_panel]:
      out_line = dict(zip(fieldnames, [None]*len(fieldnames)))
      cell_line_pair = (cell_line, cell_line) 
      if cell_line_pair not in cell_line_pair_to_spearman_tuple:
        continue     
      triplet = cell_line_pair_to_spearman_tuple[cell_line_pair]
      out_line.update({'Panel': base_panel, 'cell line': cell_line, 'num_observation': triplet[2],
        "Spearman's rho": triplet[0], 'p-value': triplet[1]})
      writer.writerow(out_line)
      out_line.update({'base cell line': cell_line})
      for compare_panel in panels_to_clines:
        if base_panel == compare_panel:
          continue
        out_line['Panel'] = compare_panel
        for compare_cline in panels_to_clines[compare_panel]:
          cell_line_pair = (compare_cline, compare_cline)
          if cell_line_pair not in cell_line_pair_to_spearman_tuple:
            continue
          compare_cline_triplet = cell_line_pair_to_spearman_tuple[cell_line_pair]
          if (cell_line, compare_cline) not in cell_line_pair_to_spearman_tuple:
            continue
          intersecting_base_triplet = cell_line_pair_to_spearman_tuple[(cell_line, compare_cline)]
          intersecting_compare_triplet = cell_line_pair_to_spearman_tuple[(compare_cline, cell_line)]          
          out_line.update({'cell line': compare_cline, 'num_observation': compare_cline_triplet[2], 
            "Spearman's rho": compare_cline_triplet[0], 'p-value': compare_cline_triplet[1], 
            'num_intersection': intersecting_base_triplet[2], 'rho-intersection': intersecting_compare_triplet[0],
            'pvalue-intersection': intersecting_compare_triplet[1], 'rho-inter-base-cline':
            intersecting_base_triplet[0], 'pvalue-inter-base-cline': intersecting_base_triplet[1]})
          writer.writerow(out_line) 
      writer.writerow(dict(zip(fieldnames, [None]*len(fieldnames))))

def get_cindex_tuple(base_panel_to_clines, panels_to_clines, panel_and_cline_to_smiles, 
  panel_cline_and_smiles_to_value, smiles_to_ar_score, base_panel='Prostate', 
  threshold = 0, smiles_to_exclude=set()):

  cell_line_pair_to_cindex_tuple = {}
  for cell_line in base_panel_to_clines[base_panel]:
    pair = (base_panel, cell_line)
    base_smiles_list = panel_and_cline_to_smiles[pair]
    base_smiles_set = set(base_smiles_list)
    if len(base_smiles_list) < threshold:
      continue
    assert len(base_smiles_set & smiles_to_exclude) == 0
    base_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(base_panel, cell_line, 
      smiles)] for smiles in base_smiles_list])
    base_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in base_smiles_list])
    cind = concordance_index(base_nci60_values, base_ar_values)
    cell_line_pair = (cell_line, cell_line)
    assert cell_line_pair not in cell_line_pair_to_cindex_tuple
    cell_line_pair_to_cindex_tuple[cell_line_pair] = (cind, len(base_nci60_values))
    for compare_panel in panels_to_clines:
      if base_panel == compare_panel:
        continue
      for compare_cline in panels_to_clines[compare_panel]:     
        this_pair = (compare_panel, compare_cline)
        compare_smiles_list = panel_and_cline_to_smiles[this_pair]
        compare_smiles_set = set(compare_smiles_list)
        assert len(compare_smiles_set & smiles_to_exclude) == 0
        intersecting_smiles_list = [smiles for smiles in base_smiles_list if smiles in 
          compare_smiles_set]
        if len(intersecting_smiles_list) < threshold:
          continue  

        cell_line_pair = (compare_cline, compare_cline) 
        if cell_line_pair not in cell_line_pair_to_cindex_tuple:          
          compare_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(compare_panel,
            compare_cline, smiles)] for smiles in compare_smiles_list])
          compare_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in compare_smiles_list])
          cind = concordance_index(compare_nci60_values, compare_ar_values)       
          cell_line_pair_to_cindex_tuple[cell_line_pair] = (cind, len(compare_nci60_values))
        cell_line_pair = (cell_line, compare_cline)
        intersecting_base_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(base_panel,
          cell_line, smiles)] for smiles in intersecting_smiles_list])
        intersecting_ar_values = np.asarray([smiles_to_ar_score[smiles] for smiles in 
          intersecting_smiles_list])
        cind = concordance_index(intersecting_base_nci60_values, intersecting_ar_values)
        cell_line_pair_to_cindex_tuple[cell_line_pair] = (cind, len(intersecting_base_nci60_values))
        cell_line_pair = (compare_cline, cell_line)
        intersecting_compare_nci60_values = np.asarray([panel_cline_and_smiles_to_value[(compare_panel,
          compare_cline, smiles)] for smiles in intersecting_smiles_list])
        cind = concordance_index(intersecting_compare_nci60_values, intersecting_ar_values)
        cell_line_pair_to_cindex_tuple[cell_line_pair] = (cind, len(intersecting_compare_nci60_values))
  return cell_line_pair_to_cindex_tuple       

def calc_cindex(pred_file, base_cell_lines=['DU-145', 'PC-3'], panels_for_comparison=['Breast'],
  base_panel='Prostate', out_file='ar_logGI50_cindex_all.csv', threshold=1500, select_subset=False,
  num_compounds=250, compound_file='ordered_compounds.csv'):
  
  df_avg = pd.read_csv(pred_file, header = 0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  if select_subset:
    df_top_compounds = pd.read_csv(compound_file, header=0, index_col=False)
    df_top_compounds = df_top_compounds.head(num_compounds)
    smiles_subset = set(df_top_compounds.loc[:, 'smiles'])
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  use_all_panels = False

  panels_to_clines = dict(zip(panels_for_comparison, [set()]*len(panels_for_comparison)))
  base_panel_to_clines = {base_panel: set()}
  panel_and_cline_to_smiles = {}
  panel_cline_and_smiles_to_value = {}
  smiles_to_ar_score = {}
  smiles_to_exclude = set()

  if len(panels_for_comparison) == 0:
    use_all_panels = True
  for row_pred in df_avg.itertuples():
    smiles = row_pred[1]
    avg_score = row_pred[4]
    assert smiles not in smiles_to_ar_score
    smiles_to_ar_score[smiles] = avg_score

  time1 = time.time()
  for row in df_nci60.itertuples():
    if np.isnan(row[7]):
      continue
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    if select_subset and smiles not in smiles_subset:
      continue
    this_panel = row[1].rstrip()
    cell_line = row[2].rstrip()
    
    if this_panel == base_panel:
      if cell_line not in base_panel_to_clines[base_panel]:
        base_panel_to_clines[base_panel].add(cell_line)
    elif use_all_panels:
      if this_panel not in panels_to_clines:
        panels_to_clines[this_panel] = set() 
      if cell_line not in panels_to_clines[this_panel]:
        panels_to_clines[this_panel].add(cell_line)
    else:
      if this_panel in panels_to_clines:
        if cell_line not in panels_to_clines[this_panel]:
          panels_to_clines[this_panel].add(cell_line)
      else:
        continue
    
    pair = (this_panel, cell_line)
    if pair not in panel_and_cline_to_smiles:
      panel_and_cline_to_smiles[pair] = []
    if smiles in set(panel_and_cline_to_smiles[pair]):
      smiles_to_exclude.add(smiles)
    panel_and_cline_to_smiles[pair].append(smiles)
    
    triplet = (this_panel, cell_line, smiles)
    panel_cline_and_smiles_to_value[triplet] = -1 * row[7]

  time2 = time.time()
  print('len(smiles_to_exclude): ', len(smiles_to_exclude))
  print('time used for iterating through NCI60 data: ', time2 - time1)
  for key in panel_and_cline_to_smiles:
    panel_and_cline_to_smiles[key] = [smiles for smiles in panel_and_cline_to_smiles[key]
      if smiles not in smiles_to_exclude]
  
  cell_line_pair_to_cindex_tuple = get_cindex_tuple(base_panel_to_clines, panels_to_clines, 
    panel_and_cline_to_smiles, panel_cline_and_smiles_to_value, smiles_to_ar_score, 
    base_panel=base_panel, threshold = threshold, smiles_to_exclude=smiles_to_exclude)

  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'cell line', 'num_observation', "cindex", 
      'num_intersection', 'cindex-intersection', 'cindex-inter-base-cline', 'base cell line']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    
    for cell_line in base_panel_to_clines[base_panel]:
      out_line = dict(zip(fieldnames, [None]*len(fieldnames)))
      cell_line_pair = (cell_line, cell_line) 
      if cell_line_pair not in cell_line_pair_to_cindex_tuple:
        continue     
      pair = cell_line_pair_to_cindex_tuple[cell_line_pair]
      out_line.update({'Panel': base_panel, 'cell line': cell_line, 'num_observation': pair[1],
        "cindex": pair[0]})
      writer.writerow(out_line)
      out_line.update({'base cell line': cell_line})
      for compare_panel in panels_to_clines:
        if base_panel == compare_panel:
          continue
        out_line['Panel'] = compare_panel
        for compare_cline in panels_to_clines[compare_panel]:
          cell_line_pair = (compare_cline, compare_cline)
          if cell_line_pair not in cell_line_pair_to_cindex_tuple:
            continue
          compare_cline_pair = cell_line_pair_to_cindex_tuple[cell_line_pair]
          if (cell_line, compare_cline) not in cell_line_pair_to_cindex_tuple:
            continue
          intersecting_base_pair = cell_line_pair_to_cindex_tuple[(cell_line, compare_cline)]
          intersecting_compare_pair = cell_line_pair_to_cindex_tuple[(compare_cline, cell_line)]          
          out_line.update({'cell line': compare_cline, 'num_observation': compare_cline_pair[1], 
            "cindex": compare_cline_pair[0], 'num_intersection': intersecting_base_pair[1], 
            'cindex-intersection': intersecting_compare_pair[0],
            'cindex-inter-base-cline': intersecting_base_pair[0]})
          writer.writerow(out_line) 
      writer.writerow(dict(zip(fieldnames, [None]*len(fieldnames))))

def plot_ar_against_gi(pred_file, cell_lines=[], panels=[], threshold=1500, 
  use_all_panels=False, select_subset=False, num_compounds=250, 
  compound_file='ordered_compounds.csv', top_compounds=False):
  
  df_avg = pd.read_csv(pred_file, header = 0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  assert top_compounds + select_subset <= 1
  if select_subset:
    df_top_compounds = pd.read_csv(compound_file, header=0, index_col=False)
    df_top_compounds = df_top_compounds.head(num_compounds)
    smiles_subset = set(df_top_compounds.loc[:, 'smiles'])
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  
  if not use_all_panels and len(cell_lines) > 0:
    panels = []
  panels_to_clines = dict(zip(panels, [set()]*len(panels)))
  panel_and_cline_to_smiles = {}
  panel_cline_and_smiles_to_value = {}
  smiles_to_ar_score = {}
  smiles_to_exclude = set()

  counter = 0
  for row_pred in df_avg.itertuples():
    smiles = row_pred[1]
    avg_score = row_pred[4]
    if top_compounds:
      if counter >= num_compounds:
        smiles_subset = set(smiles_to_ar_score)
        break
    assert smiles not in smiles_to_ar_score
    smiles_to_ar_score[smiles] = avg_score
    counter += 1

  time1 = time.time()
  for row in df_nci60.itertuples():
    if np.isnan(row[7]):
      continue
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    if (select_subset or top_compounds) and smiles not in smiles_subset:
      continue
    this_panel = row[1].rstrip()
    cell_line = row[2].rstrip()
    
    if use_all_panels:
      if this_panel not in panels_to_clines:
        panels_to_clines[this_panel] = set() 
      if cell_line not in panels_to_clines[this_panel]:
        panels_to_clines[this_panel].add(cell_line)
    elif len(cell_lines)==0:
      if this_panel in panels_to_clines and cell_line not in panels_to_clines[this_panel]:
        panels_to_clines[this_panel].add(cell_line)
    else: #Only use cell lines.
      if cell_line in cell_lines:
        if this_panel not in panels_to_clines:
          panels_to_clines[this_panel] = set()
        if cell_line not in panels_to_clines[this_panel]:
          panels_to_clines[this_panel].add(cell_line)
      else:
        continue
    
    pair = (this_panel, cell_line)
    if pair not in panel_and_cline_to_smiles:
      panel_and_cline_to_smiles[pair] = []
    if smiles in set(panel_and_cline_to_smiles[pair]):
      smiles_to_exclude.add(smiles)
    panel_and_cline_to_smiles[pair].append(smiles)
    
    triplet = (this_panel, cell_line, smiles)
    panel_cline_and_smiles_to_value[triplet] = -1 * row[7]

  time2 = time.time()
  print('len(smiles_to_exclude): ', len(smiles_to_exclude))
  print('time used for iterating through NCI60 data: ', time2 - time1)
  for key in panel_and_cline_to_smiles:
    panel_and_cline_to_smiles[key] = [smiles for smiles in panel_and_cline_to_smiles[key]
      if smiles not in smiles_to_exclude]

  for key in panel_and_cline_to_smiles:
    smiles_list = panel_and_cline_to_smiles[key]
    if len(smiles_list) < threshold:
      continue
    panel = key[0]
    cline = key[1]
    gi_value_list = [panel_cline_and_smiles_to_value[(panel, cline, cpd)] for cpd in smiles_list]
    ar_list = [smiles_to_ar_score[cpd] for cpd in smiles_list]
    # min_list = []
    # max_list = []        
    # for pair in metatask_to_task[meta_task_name]:
    #   task_ind = pair[0]
    #   y_task = y_true[:, task_ind]
    #   if self.mode == "regression":
    #     y_pred_task = y_pred[:, task_ind]
    #   else:
    #     y_pred_task = y_pred[:, task_ind, :]
    #   w_task = w[:, task_ind]
    #   y_true_task, y_pred_task = self.get_y_vectors(y_task, y_pred_task, w_task)

    #   plt.plot(y_pred_task, y_true_task, 'b.')              
    #   y_vector = np.append(y_true_task, y_pred_task)
    #   min_list.append(np.amin(y_vector))
    #   max_list.append(np.amax(y_vector))

    plt.plot(ar_list, gi_value_list, 'b.')              
    # y_vector = np.append(y_true_task, y_pred_task)
    # min_list.append(np.amin(y_vector))
    # max_list.append(np.amax(y_vector))
    # min_value = np.amin(np.array(min_list)) 
    # max_value = np.amax(np.array(max_list))         
    # plt.plot([min_value-1, max_value + 1], [min_value-1, max_value + 1], 'k') 
    plt.title("Predicted AR score vs negated logGI50 value for panel " + panel + " cell line " + cline)       
    plt.xlabel("Predicted AR antagonist scores")   
    plt.ylabel("negated logGI50 value")
    suffix = ""
    if select_subset:
      suffix += "selected_" + str(num_compounds) 
    elif top_compounds:
      suffix += "top_" + str(num_compounds) 
    image_name = "plots/ar_vs_gi/" + panel + "_" + cline + suffix + ".png"
    plt.savefig(image_name)
    plt.close()

def get_cpds_and_score_list(df_observe, smiles_set, AR_toxcast_codes=[], AR_coef_dict={},
  uniprot_ID=None, tc_code_and_protID_to_assays={}, assay_to_col_ind={}, orig_invalid_val=None,
  new_invalid_val=None):
  valid_compounds = []
  score_list = []
  for row_obs in df_observe.itertuples():
    smiles = row_obs[3]
    if smiles not in smiles_set:
      continue
    all_valid = True
    this_score = 0
    for tc_code in AR_toxcast_codes:
      if AR_coef_dict[tc_code] == 0:
        continue
      pair = (tc_code, uniprot_ID)
      assay_names = tc_code_and_protID_to_assays[pair]
      assert len(assay_names) == 1      
      value = np.nan
      for name in assay_names:
        col_index = assay_to_col_ind[name]
        value = row_obs[col_index]                           
      if np.isnan(value):
        all_valid = False
        break
      else:
        if value == orig_invalid_val:
          value = new_invalid_val
        value = 4 - np.log10(value)
        this_score += value * AR_coef_dict[tc_code]

    if all_valid:
      valid_compounds.append(smiles)
      score_list.append(this_score)

  return valid_compounds, score_list

def read_nci60(df_nci60, panel_to_use, cell_line_to_use=[], invalid_to_canon_smiles={},
  negative=True):
  panel_and_cline_to_smiles = {}
  panel_cline_and_smiles_to_value = {}
  smiles_to_exclude = set()

  time1 = time.time()
  for row in df_nci60.itertuples():    
    this_panel = row[1].rstrip()
    cell_line = row[2].rstrip()
    if this_panel != panel_to_use:
      continue
    if cell_line not in cell_line_to_use:
      continue
    if np.isnan(row[7]):
      continue
    
    smiles = row[3]
    if smiles in invalid_to_canon_smiles:
      smiles = invalid_to_canon_smiles[smiles]
    
    pair = (this_panel, cell_line)
    if pair not in panel_and_cline_to_smiles:
      panel_and_cline_to_smiles[pair] = []
    if smiles in set(panel_and_cline_to_smiles[pair]):
      smiles_to_exclude.add(smiles)
    panel_and_cline_to_smiles[pair].append(smiles)
    
    triplet = (this_panel, cell_line, smiles)
    value = row[7]
    if negative:
      value = -1 * value
    panel_cline_and_smiles_to_value[triplet] = value

  time2 = time.time()
  print('len(smiles_to_exclude): ', len(smiles_to_exclude))
  print('time used for iterating through NCI60 data: ', time2 - time1)
  for key in panel_and_cline_to_smiles:
    panel_and_cline_to_smiles[key] = [smiles for smiles in panel_and_cline_to_smiles[key]
      if smiles not in smiles_to_exclude]

  return panel_and_cline_to_smiles, panel_cline_and_smiles_to_value 

def plot_real_ar_to_gi(AR_toxcast_codes=[], plot_individually = False, 
  AR_score_coef=[], antagonist=True):
  if antagonist:
    score_type = "antagonist"
  else:
    score_type = "agonist"
  orig_invalid_val = 1000000
  new_invalid_val = 1000
  panel_to_use = "Breast"
  cell_line_to_use = ["HS 578T"]
  uniprot_ID = 'P10275'
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  tc_code_and_protID_to_assays = get_toxcast_code_dict()
  df_observe = pd.read_csv('../full_toxcast/Assays_smiles_INCHI.csv', header = 0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  
  df_observe_colnames = list(df_observe)
  ind_list = list(range(len(df_observe_colnames)))
  ind_list = [ind + 1 for ind in ind_list]
  assay_to_col_ind = dict(zip(df_observe_colnames, ind_list))
  AR_toxcast_codes = ['toxcast_' + code for code in AR_toxcast_codes]

  panel_and_cline_to_smiles, panel_cline_and_smiles_to_value = read_nci60(df_nci60, 
    panel_to_use, cell_line_to_use=cell_line_to_use, invalid_to_canon_smiles={})

  #HACK: cell_line_to_use should only contain 1 element, otherwise taking the 0th element causes problem.
  pair = (panel_to_use, cell_line_to_use[0])
  smiles_set = set(panel_and_cline_to_smiles[pair])
  
  if not plot_individually:
    AR_coef_dict = dict(zip(AR_toxcast_codes, AR_score_coef))

    valid_compounds, score_list = get_cpds_and_score_list(df_observe, smiles_set, 
      AR_toxcast_codes=AR_toxcast_codes, AR_coef_dict=AR_coef_dict, uniprot_ID=uniprot_ID, 
      orig_invalid_val=orig_invalid_val, tc_code_and_protID_to_assays=tc_code_and_protID_to_assays, 
      assay_to_col_ind=assay_to_col_ind, new_invalid_val= new_invalid_val)
    
    score_array = np.asarray(score_list)
    sorted_score_inds = np.argsort(-score_array)
    ordered_compounds = [valid_compounds[ind] for ind in sorted_score_inds]
    gi_list = [panel_cline_and_smiles_to_value[(panel_to_use, cell_line_to_use[0], smiles)] for 
      smiles in valid_compounds]
    gi_array = np.asarray(gi_list)
    sorted_gi_array = np.sort(gi_array)
    cindex = concordance_index(gi_array, score_array)
    ndcg, _, _ = ndcg_tool(ordered_compounds, panel_cline_and_smiles_to_value, sorted_gi_array, 
      cell_line=cell_line_to_use[0], panel=panel_to_use)
    rho, pval = scipy.stats.spearmanr(gi_array, score_array)
    print("Cell line: ", cell_line_to_use[0])
    print("Number of valid compounds: ", len(valid_compounds))
    print("Concordance index: ", cindex)
    print("NDCG: ", ndcg)
    print("Spearman correlation: ", rho)
    print("Spearman p-value: ", pval)

    plt.plot(score_list, gi_list, 'b.')
    plt.title(panel_to_use + " " + cell_line_to_use[0])            
    plt.xlabel("AR " + score_type + " scores")
    plt.ylabel("negated logGI50 value")
    image_name = "plots/ar_vs_gi_real/" + panel_to_use + "_" + cell_line_to_use[0] + \
      "_" + score_type + ".png"
    plt.savefig(image_name)
    plt.close()

  else:
    for i in range(len(AR_score_coef)):
      modified_score = [0.] * len(AR_score_coef)
      modified_score[i] = np.sign(AR_score_coef[i]) * 1.0
      AR_coef_dict = dict(zip(AR_toxcast_codes, modified_score))

      valid_compounds, score_list = get_cpds_and_score_list(df_observe, smiles_set, 
        AR_toxcast_codes=AR_toxcast_codes, AR_coef_dict=AR_coef_dict, uniprot_ID=uniprot_ID, 
        orig_invalid_val=orig_invalid_val, tc_code_and_protID_to_assays=tc_code_and_protID_to_assays, 
        assay_to_col_ind=assay_to_col_ind, new_invalid_val= new_invalid_val)
      
      score_array = np.asarray(score_list)
      sorted_score_inds = np.argsort(-score_array)
      ordered_compounds = [valid_compounds[ind] for ind in sorted_score_inds]
      gi_list = [panel_cline_and_smiles_to_value[(panel_to_use, cell_line_to_use[0], smiles)] for 
        smiles in valid_compounds]
      gi_array = np.asarray(gi_list)
      sorted_gi_array = np.sort(gi_array)
      cindex = concordance_index(gi_array, score_array)
      ndcg, _, _ = ndcg_tool(ordered_compounds, panel_cline_and_smiles_to_value, sorted_gi_array, 
        cell_line=cell_line_to_use[0], panel=panel_to_use)
      rho, pval = scipy.stats.spearmanr(gi_array, score_array)
      print("Cell line: ", cell_line_to_use[0])
      print("Number of valid compounds: ", len(valid_compounds))
      print("Concordance index: ", cindex)
      print("NDCG: ", ndcg)
      print("Spearman correlation: ", rho)
      print("Spearman p-value: ", pval)

      plt.plot(score_list, gi_list, 'b.')            
      prefix = ""
      if np.sign(modified_score[i]) == -1.0:
        prefix = "negated "
      plt.xlabel(prefix + AR_toxcast_codes[i])

      plt.ylabel("negated logGI50 value")
      image_name = "plots/ar_vs_gi_real/" + panel_to_use + "_" + cell_line_to_use[0] + \
        AR_toxcast_codes[i] + ".png"
      plt.savefig(image_name)
      plt.close()
 

def calculate_mean_for_observed_ar(top_n_list = [20, 10], out_file='true_ar_mean_logGI50_breast.csv', 
  threshold=30, cell_lines = ['BT-549', 'HS 578T', 'MCF7', 'MDA-MB-231/ATCC', 'T-47D'], 
  AR_toxcast_codes=[], AR_score_coef=[]):

  orig_invalid_val = 1000000
  new_invalid_val = 1000
  df_observe = pd.read_csv('../full_toxcast/Assays_smiles_INCHI.csv', header = 0, index_col=False)
  df_nci60 = pd.read_csv('NCI60_bio.csv', header=0, index_col=False)
  panel_to_use = 'Breast'
  uniprot_ID = 'P10275'
  invalid_to_canon_smiles = get_canonical_smiles_dict()
  tc_code_and_protID_to_assays = get_toxcast_code_dict()

  df_observe_colnames = list(df_observe)
  ind_list = list(range(len(df_observe_colnames)))
  ind_list = [ind + 1 for ind in ind_list]
  assay_to_col_ind = dict(zip(df_observe_colnames, ind_list))
  AR_toxcast_codes = ['toxcast_' + code for code in AR_toxcast_codes]
  AR_coef_dict = dict(zip(AR_toxcast_codes, AR_score_coef))

  panel_and_cline_to_smiles, panel_cline_and_smiles_to_value = read_nci60(df_nci60, 
    panel_to_use, cell_line_to_use=cell_lines, invalid_to_canon_smiles={}, negative=False)

  cline_to_topn_list = {}
  cline_and_topn_to_value_list = {}

  for cell_line in cell_lines:
    pair = (panel_to_use, cell_line)
    smiles_set = set(panel_and_cline_to_smiles[pair])

    valid_compounds, score_list = get_cpds_and_score_list(df_observe, smiles_set, 
      AR_toxcast_codes=AR_toxcast_codes, AR_coef_dict=AR_coef_dict, uniprot_ID=uniprot_ID, 
      orig_invalid_val=orig_invalid_val, tc_code_and_protID_to_assays=tc_code_and_protID_to_assays, 
      assay_to_col_ind=assay_to_col_ind, new_invalid_val= new_invalid_val)

    score_array = np.asarray(score_list)
    sorted_score_inds = np.argsort(-score_array)
    ordered_compounds = [valid_compounds[ind] for ind in sorted_score_inds]

    size = len(score_list)
    if size < threshold:
      continue
    cline_top_n_list = [size] + top_n_list    
    cline_to_topn_list[cell_line] = cline_top_n_list

    for i, top_n in enumerate(cline_top_n_list):
      if top_n > size:
        continue
      if i < len(cline_top_n_list) - 1:
        # Sanity check.
        assert cline_top_n_list[i + 1] <= top_n
      compound_to_value_subset = {} 
      for smiles in ordered_compounds:  
        triplet = (panel_to_use, cell_line, smiles)      
        compound_to_value_subset[smiles] = panel_cline_and_smiles_to_value[triplet]  
        if len(compound_to_value_subset) >= top_n:
          cline_and_topn_to_value_list[(cell_line, top_n)] = list(compound_to_value_subset.values())
          break
  
  with open(out_file, 'w', newline='') as csvfile:
    fieldnames = ['Panel', 'top_n', 'cell line', 'num_observation', 'mean value', 
      'standard deviation']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    out_line = {'Panel': panel_to_use}
    for cell_line in cell_lines:
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
          'standard deviation': np.std(values, ddof=1)})
        writer.writerow(out_line)

if __name__ == "__main__":
  dataset = 'toxcast'
  #dataset = 'kiba'
  # produce_dataset(dataset_used=dataset, prot_desc_path_list=['../full_toxcast/prot_desc.csv'], 
  #   get_all_compounds=True, take_mol_subset=False, prot_pairs_to_choose=AR_list_s + ER_list_s)
  #produce_dataset(dataset_used=dataset, output_prefix="all_prot_intxn")
  # synthesize_ranking('preds_arer_tc_ecfp.csv', 'ordered_arer_tc_ecfp_6620.csv', 
  #   AR_toxcast_codes=AR_toxcast_codes, dataset_used=dataset, AR_score_coef=AR_antagonist_score_coef)   
  # synthesize_ranking('preds_arer_tc_graphconv_oversp.csv', 'ordered_arer_tc_gc_oversp.csv', 
  #   direction=True, dataset_used=dataset, AR_toxcast_codes=AR_toxcast_codes, 
  #   AR_score_coef=AR_antagonist_score_coef)
  # compare('avg_ar_tc.csv', 'avg_ar_tc_oversp.csv', cutoff=100, 
  #   exclude_prot=ER_list_s)
  #get_invalid_smiles(out_file = 'invalid_smiles.csv')  
  # get_avg(input_files_list=['ordered_arer_tc_ecfp_6620.csv', 'ordered_arer_tc_gc_6620.csv'], 
  #   output_file_name = 'avg_ar_tc_6620.csv', exclude_prot=ER_list_s)

  # get_highest_similarity(input_file='avg_ar_tc_6620.csv', output_file='avg_ar_tc_6620_similarity_rdkit.csv',
  #   comparison_file='../full_toxcast/restructured.csv', num_compounds=1800)
  # filter_similarity(input_file='avg_ar_tc_6620_similarity_rdkit.csv', 
  #   output_file='avg_ar_tc_6620_sim_rdkit_subset.csv', draw_2d_image=False, threshold=0.98, limit=False, limit_lines=10)
  #calculate_mean_activity('avg_ar_tc.csv')
  #calculate_ndcg('avg_ar_tc.csv')
  #calc_subsets_cindex('avg_ar_tc.csv')
  #plot_values()
  #get_intersection()
  # get_selective_compounds()
  # calc_spearmanr('avg_ar_tc.csv', panels_for_comparison=[], threshold=30, select_subset=True,
  #   compound_file='ordered_compounds_2.csv')
  #calc_cindex('avg_ar_tc.csv', panels_for_comparison=[], compound_file='ordered_compounds.csv')
  # plot_ar_against_gi('avg_ar_tc.csv', panels=['Central Nervous System', 'Leukemia'], threshold=100, 
  #   top_compounds=True, num_compounds=1000)
  # plot_real_ar_to_gi(AR_toxcast_codes=AR_toxcast_codes, plot_individually=False, 
  #   AR_score_coef=AR_antagonist_score_coef, antagonist=True)
  # calculate_mean_for_observed_ar(AR_toxcast_codes=AR_toxcast_codes, 
  #   AR_score_coef=AR_antagonist_score_coef)