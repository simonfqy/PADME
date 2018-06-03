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
import re
import deepchem
from deepchem.trans import undo_transforms
import dcCustom

# This script is intended to parse the dataset generated in PADME(dcCustom) to a readable form
# of SimBoost and KronRLS.

def parse_data(dataset_nm='davis', featurizer = 'GraphConv', split='random', K = 5, 
  mode = 'regression', predict_cold = False, cold_drug=False, cold_target=False, 
  split_warm=False, filter_threshold=0, create_mapping=False):

  assert (predict_cold + cold_drug + cold_target + split_warm) <= 1
  if mode == 'regression' or mode == 'reg-threshold':
    mode = 'regression'

  smiles_to_some_id = {}
  duplicated_drugs = set()
  some_id_name = 'cid'
  cmpd_file_name = "compound_cids.txt"
  if re.search('davis', dataset_nm, re.I):
    data_dir = "davis_data/"    
    with open(data_dir + "SMILES_CIDs_corrected.txt", 'r') as f:
      data = f.readlines()
      for line in data:        
        words = line.split()        
        if words[1] not in smiles_to_some_id:
          smiles_to_some_id[words[1]] = words[0]

  elif re.search('metz', dataset_nm, re.I):
    data_dir = "metz_data/"
    some_id_name = 'sid'
    cmpd_file_name = "compound_sids.txt"
    df = pd.read_csv(data_dir + 'Metz_interaction.csv', header = 0, index_col=0, usecols=range(3))
    for row in df.itertuples():         
      if row[2] != row[2]:
        continue
      if row[2] not in smiles_to_some_id:
        smiles_to_some_id[row[2]] = str(int(row[1]))        
      
    print("length of dictionary smiles_to_some_id: ", len(smiles_to_some_id))

  elif re.search('kiba', dataset_nm, re.I):
    data_dir = "KIBA_data/"
    some_id_name = "CHEMBL_ID"
    cmpd_file_name = "Chembl_ids.txt"
    df = pd.read_csv(data_dir + 'Smiles_bio_results.csv', header = 0, index_col=0, usecols=range(2))
    for row in df.itertuples():
      if row[1] != row[1]:
        continue
      if row[1] not in smiles_to_some_id:
        smiles_to_some_id[row[1]] = row[0]

  # elif re.search('toxcast', dataset_nm, re.I):
  #   data_dir = "full_toxcast/"
  #   some_id_name = "InchiKey"
  #   cmpd_file_name = "Inchikeys.txt"
  #   df = pd.read_csv(data_dir + 'Assays_smiles_INCHI.csv', header = 0, index_col=3, 
  #     usecols=list(range(1, 5))+[14])
  #   for row in df.itertuples():
  #     if row[2] != row[2]:
  #       continue
  #     if row[2] not in smiles_to_some_id:
  #       smiles_to_some_id[row[2]] = row[4]
  simboost_data_dir = "./SimBoost/data/" + data_dir

  suffix = ""
  if filter_threshold > 0:
    suffix = "_filtered" + suffix
  if predict_cold:
    suffix = "_cold" + suffix
  elif split_warm:
    suffix = "_warm" + suffix
  elif cold_drug:
    suffix = "_cold_drug" + suffix
  elif cold_target:
    suffix = "_cold_target" + suffix
  
  featurizer = featurizer + "_CV" 
  save_dir = os.path.join(data_dir, featurizer + suffix + "/" + mode + "/" + split)
  loaded, all_dataset, transformers = dcCustom.utils.save.load_cv_dataset_from_disk(
      save_dir, K)
  
  assert loaded
  pair_to_fold = {}
  pair_to_value = {}
  dataset_length = 0
  time_start = time.time()
  if not create_mapping:
    # Use the existing mappings stored in prot_info.csv and drug_info.csv.
    drug_pair_mapping = {}
    prot_pair_mapping = {}
    drug_pair_list = []
    prot_pair_list = []
    df_drug = pd.read_csv(data_dir + 'drug_info.csv', header = 0, index_col=0)
    for row in df_drug.itertuples():
      pair = (str(row[0]), row[1])
      if pair not in drug_pair_mapping:
        drug_pair_mapping[pair] = row[2]
        drug_pair_list.append(pair)
    
    df_prot = pd.read_csv(data_dir + 'prot_info.csv', header = 0, index_col=0)
    for row in df_prot.itertuples():
      pair = (row[0], row[1])
      if pair not in prot_pair_mapping:
        prot_pair_mapping[pair] = row[2]
        prot_pair_list.append(pair)         
  else:
    drug_mapping = {}
    prot_mapping = {}
    drug_list = []
    prot_list = []

  for i in range(K):
    train_data = all_dataset[i][0]
    validation_data = all_dataset[i][1]
    
    for (X_b, y_b, w_b, _) in validation_data.itersamples():      
      assert w_b[0] == 1.0
      drug_mol = X_b[0]
      protein = X_b[1]
      y_b = undo_transforms(y_b, transformers)
      if not create_mapping:
        drug_smiles = drug_mol.smiles
        some_id = smiles_to_some_id[drug_smiles]
        drug_pair = (some_id, drug_smiles)
        drug_ind = drug_pair_mapping[drug_pair]
        prot_seq = protein.get_sequence()[1]
        prot_name = protein.get_name()[1]
        prot_pair = (prot_name, prot_seq)
        prot_ind = prot_pair_mapping[prot_pair]

        pair = (drug_ind, prot_ind)
        assert pair not in pair_to_fold        
        pair_to_fold[pair] = i + 1      
        if pair in pair_to_value:
          assert pair_to_value[pair] == y_b[0]
        else:       
          pair_to_value[pair] = y_b[0]        
        dataset_length += 1

      else:
        if drug_mol not in drug_mapping:
          # values start from 1.
          num_drug = len(drug_mapping)
          drug_mapping[drug_mol] = num_drug + 1
          drug_list.append(drug_mol)
          
        if protein not in prot_mapping:
          num_prot = len(prot_mapping)
          prot_mapping[protein] = num_prot + 1
          prot_list.append(protein)

        pair = (drug_mol, protein)
        assert pair not in pair_to_fold
        # Also start from 1.
        pair_to_fold[pair] = i + 1      
        if pair in pair_to_value:
          assert pair_to_value[pair] == y_b[0]
        else:
          assert i == 0        
          pair_to_value[pair] = y_b[0]

        if i == 0:
          dataset_length += 1

    if i == 0 and create_mapping:
      for (X_b, y_b, w_b, _) in train_data.itersamples():
        assert w_b[0] == 1.0
        drug_mol = X_b[0]
        protein = X_b[1]
        if drug_mol not in drug_mapping:
          num_drug = len(drug_mapping)
          drug_mapping[drug_mol] = num_drug + 1
          
        if protein not in prot_mapping:
          num_prot = len(prot_mapping)
          prot_mapping[protein] = num_prot + 1
        pair = (drug_mol, protein)
        if pair not in pair_to_value:
          y_b = undo_transforms(y_b, transformers)
          pair_to_value[pair] = y_b[0]
        dataset_length += 1

  assert len(pair_to_fold) == dataset_length
  # Now we need to construct a compound_cids.txt file according to the order in drug_list.
  if create_mapping:
    print("len(drug_list): ", len(drug_list))    
    with open(data_dir + cmpd_file_name, 'w') as f:
      some_id_list = []
      for drug_mol in drug_list:
        drug_smiles = drug_mol.smiles
        some_id = smiles_to_some_id[drug_smiles]
        some_id_list.append(some_id)
      f.write('\n'.join(some_id_list))
      sfile = open(data_dir + cmpd_file_name, 'w')
      sfile.write('\n'.join(some_id_list))
      sfile.close()
  else:
    print("len(drug_pair_list): ", len(drug_pair_list))    

  dirs = [data_dir, simboost_data_dir]
  # Every time, write twice, in two different directories respectively.
  for directory in dirs:
    if not create_mapping:
      continue
    with open(directory + "prot_info.csv", 'w', newline='') as csvfile:
      fieldnames = ['name', 'sequence', 'index']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      for protein in prot_list:
        prot_seq = protein.get_sequence()[1]
        prot_name = protein.get_name()[1]
        ind = prot_mapping[protein]
        writer.writerow({'name': prot_name, 'sequence': prot_seq, 'index': ind})

  for directory in dirs:
    if not create_mapping:
      continue
    with open(directory + "drug_info.csv", 'w', newline='') as csvfile:
      fieldnames = [some_id_name, 'smiles', 'index']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      for drug_mol in drug_list:
        drug_smiles = drug_mol.smiles
        some_id = smiles_to_some_id[drug_smiles]
        drug_ind = drug_mapping[drug_mol]
        writer.writerow({some_id_name: some_id, 'smiles': drug_smiles, 'index': drug_ind})

  triplet_split_name = "triplet_split" + suffix + ".csv"
  for directory in dirs:
    with open(directory + triplet_split_name, 'w', newline='') as csvfile:
      fieldnames = ['drug', 'target', 'value', 'fold']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      if create_mapping:
        for drug_mol in drug_list:
          drug_ind = drug_mapping[drug_mol]
          for protein in prot_list:        
            pair = (drug_mol, protein)
            if pair not in pair_to_value:
              assert pair not in pair_to_fold
              continue
            prot_ind = prot_mapping[protein]
            value = pair_to_value[pair]
            fold_ind = pair_to_fold[pair]
            writer.writerow({'drug': drug_ind, 'target': prot_ind, 'value': value,
              'fold': fold_ind})
      else:
        for drug_pair in drug_pair_list:
          drug_ind = drug_pair_mapping[drug_pair]
          for prot_pair in prot_pair_list:
            prot_ind = prot_pair_mapping[prot_pair]        
            pair = (drug_ind, prot_ind)
            if pair not in pair_to_value:
              assert pair not in pair_to_fold
              continue            
            value = pair_to_value[pair]
            fold_ind = pair_to_fold[pair]
            writer.writerow({'drug': drug_ind, 'target': prot_ind, 'value': value,
              'fold': fold_ind})

  time_end = time.time()
  print("Processing took %f seconds." % (time_end - time_start))

if __name__ == '__main__':
  #parse_data(featurizer='ECFP', cold_target=True, filter_threshold=1)
  parse_data(dataset_nm='metz', featurizer = 'ECFP', cold_drug=True, filter_threshold=1)
  #parse_data(dataset_nm='kiba', featurizer = 'ECFP', split_warm=True, filter_threshold=6)
  
