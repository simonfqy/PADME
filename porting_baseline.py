from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd
import os
import time
import pdb
import csv
import re
from dcCustom.trans import undo_transforms
import dcCustom
from collections import OrderedDict

# This script is intended to parse the dataset generated in PADME(dcCustom) to a readable form
# of SimBoost and KronRLS.

def get_pair_values_and_fold_ind(all_dataset, K, transformers, create_mapping=False, 
  smiles_to_some_id=None, drug_id_and_smiles_to_ind=None, prot_name_and_seq_to_ind=None, 
  dt_pair_to_fold=None, dt_pair_to_value=None, drug_mol_to_ind = None, prot_to_ind=None):  
  
  for i in range(K):
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
        drug_ind = drug_id_and_smiles_to_ind[drug_pair]
        prot_seq = protein.get_sequence()[1]
        prot_name = protein.get_name()[1]
        prot_pair = (prot_name, prot_seq)
        prot_ind = prot_name_and_seq_to_ind[prot_pair]
        pair = (drug_ind, prot_ind)

      else:
        if drug_mol not in drug_mol_to_ind:
          # values start from 1.
          drug_mol_to_ind[drug_mol] = len(drug_mol_to_ind) + 1
          
        if protein not in prot_to_ind:
          prot_to_ind[protein] = len(prot_to_ind) + 1
        pair = (drug_mol, protein)
      
      assert pair not in dt_pair_to_fold
      # Also start from 1.
      dt_pair_to_fold[pair] = i + 1      
      assert pair not in dt_pair_to_value
      dt_pair_to_value[pair] = y_b[0]
  

def parse_data(dataset_nm='davis', featurizer = 'GraphConv', split='random', K = 5, 
  mode = 'regression', predict_cold = False, cold_drug=False, cold_target=False, 
  split_warm=False, cold_drug_cluster=False, filter_threshold=0, create_mapping=False, 
  input_protein=True):

  assert (predict_cold + cold_drug + cold_target + split_warm + cold_drug_cluster) <= 1
  if mode == 'regression' or mode == 'reg-threshold':
    mode = 'regression'

  smiles_to_some_id = {}
  duplicated_drugs = set()
  some_id_name = 'cid'
  cmpd_file_name = "compound_cids.txt"
  prot_file_name = "prot_info.csv"
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

  simboost_data_dir = "./SimBoost/data/" + data_dir

  suffix = ""
  if not input_protein:
    suffix = "_no_prot" + suffix
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
  elif cold_drug_cluster:
    suffix = "_cold_drug_cluster" + suffix

  if re.match('GraphConv', featurizer, re.I):
    opt_suffix = "_gc"
  elif re.match('Weave', featurizer, re.I):
    opt_suffix = "_wv"
  elif re.match('ecfp', featurizer, re.I):
    opt_suffix = ""
  else:
    assert False
  
  featurizer = featurizer + "_CV" 
  save_dir = os.path.join(data_dir, featurizer + suffix + "/" + mode + "/" + split)
  loaded, all_dataset, transformers = dcCustom.utils.save.load_cv_dataset_from_disk(
      save_dir, K)
  
  assert loaded
  dt_pair_to_fold = {}
  dt_pair_to_value = {}
  drug_id_and_smiles_to_ind = OrderedDict()
  prot_name_and_seq_to_ind = OrderedDict()
  time_start = time.time()
  if not create_mapping:
    # Use the existing mappings stored in prot_info.csv and drug_info.csv.
    df_drug = pd.read_csv(data_dir + 'drug_info.csv', header = 0, index_col=0)
    for row in df_drug.itertuples():
      pair = (str(row[0]), row[1])
      if pair not in drug_id_and_smiles_to_ind:
        drug_id_and_smiles_to_ind[pair] = row[2]        
    
    df_prot = pd.read_csv(data_dir + 'prot_info.csv', header = 0, index_col=0)
    for row in df_prot.itertuples():
      pair = (row[0], row[1])
      if pair not in prot_name_and_seq_to_ind:
        prot_name_and_seq_to_ind[pair] = row[2]             

  drug_mol_to_ind = OrderedDict()
  prot_to_ind = OrderedDict()

  if input_protein:
    get_pair_values_and_fold_ind(all_dataset, K, transformers, create_mapping=create_mapping, 
      smiles_to_some_id=smiles_to_some_id, drug_id_and_smiles_to_ind=drug_id_and_smiles_to_ind, 
      prot_name_and_seq_to_ind=prot_name_and_seq_to_ind, dt_pair_to_fold=dt_pair_to_fold,
      dt_pair_to_value=dt_pair_to_value, drug_mol_to_ind = drug_mol_to_ind, prot_to_ind=prot_to_ind)  
  else:
    raise ValueError("Currently input_protein==False scenario is unsupported.")
  
  # Now we need to construct a compound_cids.txt file according to the order in drug_mol_to_ind.
  if create_mapping:
    print("len(drug_mol_to_ind): ", len(drug_mol_to_ind))    
    with open(data_dir + cmpd_file_name, 'w') as f:
      some_id_list = []
      for drug_mol in drug_mol_to_ind:
        drug_smiles = drug_mol.smiles
        some_id = smiles_to_some_id[drug_smiles]
        some_id_list.append(some_id)
      f.write('\n'.join(some_id_list))
      sfile = open(data_dir + cmpd_file_name, 'w')
      sfile.write('\n'.join(some_id_list))
      sfile.close()
  else:
    print("len(drug_id_and_smiles_to_ind): ", len(drug_id_and_smiles_to_ind))    

  dirs = [data_dir, simboost_data_dir]
  # Every time, write twice, in two different directories respectively.
  for directory in dirs:
    if not create_mapping:
      continue
    with open(directory + prot_file_name, 'w', newline='') as csvfile:
      fieldnames = ['name', 'sequence', 'index']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      for protein, ind in prot_to_ind.items():
        prot_seq = protein.get_sequence()[1]
        prot_name = protein.get_name()[1]
        writer.writerow({'name': prot_name, 'sequence': prot_seq, 'index': ind})

  for directory in dirs:
    if not create_mapping:
      continue
    with open(directory + "drug_info.csv", 'w', newline='') as csvfile:
      fieldnames = [some_id_name, 'smiles', 'index']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      for drug_mol, drug_ind in drug_mol_to_ind.items():
        drug_smiles = drug_mol.smiles
        some_id = smiles_to_some_id[drug_smiles]
        writer.writerow({some_id_name: some_id, 'smiles': drug_smiles, 'index': drug_ind})

  suffix = suffix + ""
  triplet_split_name = "triplet_split" + opt_suffix + suffix + ".csv"
  for directory in dirs:
    with open(directory + triplet_split_name, 'w', newline='') as csvfile:
      fieldnames = ['drug', 'target', 'value', 'fold']
      writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
      writer.writeheader()
      if create_mapping:
        for drug_mol, drug_ind in drug_mol_to_ind.items():
          for protein, prot_ind in prot_to_ind.items():        
            pair = (drug_mol, protein)
            if pair not in dt_pair_to_value:
              assert pair not in dt_pair_to_fold
              continue
            value = dt_pair_to_value[pair]
            fold_ind = dt_pair_to_fold[pair]
            writer.writerow({'drug': drug_ind, 'target': prot_ind, 'value': value,
              'fold': fold_ind})
      else:
        for _, drug_ind in drug_id_and_smiles_to_ind.items():
          for _, prot_ind in prot_name_and_seq_to_ind.items():  
            pair = (drug_ind, prot_ind)
            if pair not in dt_pair_to_value:
              assert pair not in dt_pair_to_fold
              continue            
            value = dt_pair_to_value[pair]
            fold_ind = dt_pair_to_fold[pair]
            writer.writerow({'drug': drug_ind, 'target': prot_ind, 'value': value,
              'fold': fold_ind})

  time_end = time.time()
  print("Processing took %f seconds." % (time_end - time_start))

if __name__ == '__main__':
  # parse_data(featurizer='GraphConv', cold_target=True, filter_threshold=1)
  # parse_data(featurizer='GraphConv', cold_target=True, filter_threshold=1)
  parse_data(dataset_nm='kiba', featurizer='ECFP', cold_drug_cluster=True, filter_threshold=6)
  # parse_data(dataset_nm='kiba', featurizer='GraphConv', cold_drug_cluster=True, filter_threshold=6)
  # parse_data(featurizer='GraphConv', split_warm=True, filter_threshold=1)
  # parse_data(dataset_nm='metz', featurizer='GraphConv', split_warm=True, filter_threshold=1)
  # parse_data(dataset_nm='metz', featurizer='GraphConv', cold_drug=True, filter_threshold=1)
  # parse_data(dataset_nm='metz', featurizer='GraphConv', cold_target=True, filter_threshold=1)
  #parse_data(dataset_nm='kiba', cold_target=True, filter_threshold=6)
  # parse_data(dataset_nm='kiba', featurizer='GraphConv', split_warm=True, filter_threshold=6)
  # parse_data(dataset_nm='kiba', featurizer='GraphConv', cold_drug=True, filter_threshold=6)
  # parse_data(dataset_nm='kiba', featurizer='GraphConv', cold_target=True, filter_threshold=6)
  
