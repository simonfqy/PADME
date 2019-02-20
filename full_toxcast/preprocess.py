from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import pdb
import csv
import math
import matplotlib.pyplot as plt
import time

EPSILON = 1e-4

df_uniprot = pd.read_csv('Assay_UniprotID_subgroup.csv', header = 0, index_col=0)
subgroup_dict = dict(df_uniprot.iloc[:]['Subgroup'])
protID_dict = dict(df_uniprot.iloc[:]['Uniprot_ID'])
protList = list(set(df_uniprot.iloc[:]['Uniprot_ID']))
num_subgroups = len(set(df_uniprot.iloc[:]['Subgroup']))

df_mols = pd.read_csv('mol.csv', header = 0, index_col=False)
mol_mapping = {}
for i in range(len(df_mols)):
  mol_mapping[df_mols.iloc[i][0]] = df_mols.iloc[i][1]


def generate_data(input_csv, binarize=False, head_only=False, head_row_num=300, 
  limit_rows=False, limit_row_num=2400, prefix="toxcast_", input_prot=True, output_csv=None):
  
  df = pd.read_csv(input_csv, header = 0, index_col=2)
  if head_only:
    df=df.head(head_row_num)
  assayList = list(df)[16:]
  molList = []
  for mol in list(df.index):
    if mol != mol:
      continue
    if mol in mol_mapping:
      mol = mol_mapping[mol]
    molList.append(mol)    
    
  row_ind = 0
  col_ind = 0
  INACTIVE_VAL = 1000.0
  min_val = 1000
  max_value = 0
  inactive_counter = 0
  num_pairs = len(molList) * len(protList)
  if input_prot:
    data_matrix = np.empty((num_pairs, num_subgroups))
  else:
    data_matrix = np.empty((len(molList), len(assayList)))
  data_matrix[:] = np.nan
  pair_list = []
  subgroup_list = []
  smiles_set = set()
  pair_set = set()
  subgroup_set = set()
  # The (compound, protein) pair or group id are keys, and row or column indices are values.
  pair_to_row_ind = {}
  group_to_col_ind = {}
  na_counter = 0
  time_start_loop = time.time()
  for j in range(len(df)):
    this_row = dict(df.iloc[j])
    assert df.index[j] == df.index[j]
    smiles = molList[j]
    assert smiles not in smiles_set
    smiles_set.add(smiles)
    if j % 400 == 1:
      print("Now processing dataframe row: ", j + 1)
    for i, assay_name in enumerate(assayList):
      entry_value = this_row[assay_name]
      if entry_value != entry_value:
        na_counter += 1
        continue
      if entry_value == 1000000:
          entry_value = INACTIVE_VAL
          inactive_counter += 1
      elif entry_value > max_value:
        max_value = entry_value

      if entry_value < min_val and entry_value != 0:
        min_val = entry_value
      
      if input_prot:
        prot_id = protID_dict[assay_name]
        group_id = subgroup_dict[assay_name]
        pair = (smiles, prot_id)
        
        if pair not in pair_set:        
          pair_to_row_ind[pair] = len(pair_list)
          pair_list.append(pair)
          pair_set.add(pair)
                   
        row_ind = pair_to_row_ind[pair]

        if group_id not in subgroup_set:
          group_to_col_ind[group_id] = len(subgroup_list)
          subgroup_list.append(group_id)
          subgroup_set.add(group_id)        
                
        col_ind = group_to_col_ind[group_id]

        assert data_matrix[row_ind, col_ind] != data_matrix[row_ind, col_ind]
        data_matrix[row_ind, col_ind] = entry_value        
        
      else:
        assert data_matrix[j, i] != data_matrix[j, i]
        data_matrix[j, i] = entry_value
  
  time_end_loop = time.time()
  print("Time used in the loop: ", time_end_loop - time_start_loop)
  print("Minimum value: ", min_val)
  if input_prot:
    data_matrix = data_matrix[:len(pair_list), :len(subgroup_list)]
  # Now we have: na_counter: 4124417, inactive_counter: 954536, goodval_counter (already deleted): 66551.

  data_matrix = 4 - np.log10(data_matrix)
  max_val = 4 - np.log10(min_val)
  inf_indices = np.where(np.isinf(data_matrix))
  #pdb.set_trace()
  num_inf_entries = len(inf_indices[0])
  for i in range(num_inf_entries):
    first_ind = inf_indices[0][i]
    second_ind = inf_indices[1][i]
    assert np.isinf(data_matrix[first_ind, second_ind])
    data_matrix[first_ind, second_ind] = np.ceil(max_val)

  # num_bins = 20
  # fig, ax = plt.subplots()
  # #n, bins, patches = ax.hist(interactions, num_bins, density=1)
  # interaction_flattened = data_matrix.flatten()
  # interaction_flattened = interaction_flattened[~np.isnan(interaction_flattened)]
  # ax.hist(interaction_flattened, num_bins)
  # #ax.hist(interaction_flattened, num_bins, range=(1.01, np.ceil(max_val)))
  # ax.set_xlabel('ToxCast assay values')
  # ax.set_ylabel('Occurrence')
  # ax.set_title('Histogram of ToxCast assay values')
  # #ax.plot(bins)
  # #fig.tight_layout()
  # #plt.show()
  # #pdb.set_trace()

  #counter = 0
  shuffled_pairs = np.random.permutation(range(len(pair_list)))
  start_writing = time.time()    
  with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['smiles']
    if input_prot:
      subgroup_list = [prefix + str(task_id) for task_id in subgroup_list]
      fieldnames = subgroup_list + fieldnames + ['proteinName', 'protein_dataset']
    else:
      col_names = [prefix + str(subgroup_dict[assay_name]) + '_' + assay_name for assay_name in assayList]
      fieldnames = col_names + fieldnames
      
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    if input_prot:
      for i in range(len(pair_list)):
        if limit_rows and i > limit_row_num - 1:
          break
        index = shuffled_pairs[i]
        pair = pair_list[index]
        data_line = ['' if np.isnan(entry) else entry for entry in data_matrix[index]]
        line_values = dict(zip(subgroup_list, data_line))
        out_line = {'smiles': pair[0], 'proteinName': pair[1], 'protein_dataset': 'toxcast'}
        line_values.update(out_line)
        writer.writerow(line_values)
        
    else:
      shuffled_mol_list = np.random.permutation(range(len(molList)))      

      for ind in range(len(molList)):
        if limit_rows and ind > limit_row_num - 1:
          break
        i = shuffled_mol_list[ind]
        output_dict = {'smiles': molList[i]}
        for j, assay_name in enumerate(assayList):
          intxn_value = data_matrix[i, j]
          if intxn_value != intxn_value:
            intxn_value = ''
          col_name = prefix + str(subgroup_dict[assay_name]) + '_' + assay_name
          output_dict[col_name] = intxn_value
        writer.writerow(output_dict)

  end_writing = time.time()
  print("Time spent in writing: ", end_writing - start_writing)

def is_active(interaction_array, direction, inactive_threshold):
  if not direction:
    interaction_array = -interaction_array
    inactive_threshold = -inactive_threshold
  for element in interaction_array:
    if element == element and element > inactive_threshold:
      return True
  return False

def oversample(input_file, output_file, direction=True, inactive_threshold = 1.0+EPSILON):
  df_input = pd.read_csv(input_file, header = 0, index_col=False)
  len_input = len(df_input)
  headers = list(df_input)
  pairs_to_interaction = {}
  inactive_pair_list = []
  active_pair_list = []
  for i in range(len_input):
    this_row = df_input.iloc[i]
    smiles = this_row['smiles']
    protein_name = this_row['proteinName']
    pair = (smiles, protein_name)
    interaction_profile = np.array(this_row[:len(headers)-3])
    active = is_active(interaction_profile, direction, inactive_threshold)
    if active:
      active_pair_list.append(pair)
    else:
      inactive_pair_list.append(pair)
    pairs_to_interaction[pair] = interaction_profile
  assert len(active_pair_list) < len(inactive_pair_list)
  difference = len(inactive_pair_list) - len(active_pair_list)
  oversampled_pairs = np.random.choice(len(active_pair_list), difference)
  increased_active_pairs_ind = np.concatenate((list(range(len(active_pair_list))), 
    oversampled_pairs))
  
  shuffled_inactive = np.random.permutation(range(len(inactive_pair_list)))
  shuffled_active = np.random.permutation(range(len(increased_active_pairs_ind)))
  start_writing = time.time()
  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = headers
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    inactive_index = 0
    active_index = 0

    while True:
      inactive_remain = len(inactive_pair_list) - inactive_index
      active_remain = len(increased_active_pairs_ind) - active_index
      if inactive_remain <= 0 and active_remain <= 0:
        break
      rand_num = np.random.sample()
      if rand_num <= inactive_remain/(inactive_remain + active_remain):
        shuffled_ind = shuffled_inactive[inactive_index]
        current_pair = inactive_pair_list[shuffled_ind]
        inactive_index += 1
      else:
        shuffled_ind = shuffled_active[active_index]
        current_pair = active_pair_list[increased_active_pairs_ind[shuffled_ind]]
        active_index += 1
      #pdb.set_trace()
      interaction_profile = pairs_to_interaction[current_pair]
      data_line = ['' if np.isnan(entry) else entry for entry in interaction_profile]
      line_values = dict(zip(headers[:-3], data_line))
      out_line = {'smiles': current_pair[0], 'proteinName': current_pair[1], 
        'protein_dataset': 'toxcast'}
      line_values.update(out_line)
      writer.writerow(line_values)

  end_writing = time.time()
  print("Time spent in writing: ", end_writing - start_writing)  


if __name__ == "__main__":
  generate_data('Assays_smiles_INCHI.csv', input_prot=False, output_csv='restructured_no_prot.csv')
  # oversample('restructured.csv', 'restructured_oversampled.csv')
  