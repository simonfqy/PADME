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

df_uniprot = pd.read_csv('Assay_UniprotID_subgroup.csv', header = 0, index_col=0)
subgroup_dict = dict(df_uniprot.iloc[:]['Subgroup'])
protID_dict = dict(df_uniprot.iloc[:]['Uniprot_ID'])
protList = list(set(df_uniprot.iloc[:]['Uniprot_ID']))
num_subgroups = len(set(df_uniprot.iloc[:]['Subgroup']))

df = pd.read_csv('Assays_smiles_INCHI.csv', header = 0, index_col=2)
#df=df.head(300)
assayList = list(df)[16:]
molList = list(df.index)
molList = [mol for mol in molList if mol==mol]
pair_dict = {}
smiles_indices = {}
invalid_mols = set()
duplicate_mols = set()
row_ind = 0
col_ind = 0
INACTIVE_VAL = 1000.0
min_val = 1000
max_value = 0
inval_counter = 0
num_pairs = len(molList) * len(protList)
data_matrix = np.empty((num_pairs, num_subgroups))
data_matrix[:] = np.nan
pair_list = []
subgroup_list = []
pair_set = set()
subgroup_set = set()
# The the (compound, protein) pair or group id are keys, and row or column indices are values.
row_ind_dict = {}
col_ind_dict = {}
na_counter = 0
goodval_counter = 0
time_start_loop = time.time()
for j in range(len(df)):
  this_row = dict(df.iloc[j])
  assert df.index[j] == df.index[j]
  smiles = molList[j]
  if j % 400 == 1:
    print("Now processing dataframe row: ", j + 1)
  for i, assay_name in enumerate(assayList):
    entry_value = this_row[assay_name]
    if entry_value != entry_value:
      na_counter += 1
      continue
    prot_id = protID_dict[assay_name]
    group_id = subgroup_dict[assay_name]
    pair = (smiles, prot_id)

    if entry_value == 1000000:
      entry_value = INACTIVE_VAL
      inval_counter += 1
    elif entry_value > max_value:
      max_value = entry_value

    if entry_value < min_val and entry_value != 0:
      min_val = entry_value

    if pair not in pair_set:
      pair_list.append(pair)
      pair_set.add(pair)
      row_ind_dict[pair] = len(pair_list) - 1
         
    row_ind = row_ind_dict[pair]

    if group_id not in subgroup_set:
      subgroup_list.append(group_id)
      subgroup_set.add(group_id)
      col_ind_dict[group_id] = len(subgroup_list) - 1
            
    col_ind = col_ind_dict[group_id]

    assert data_matrix[row_ind, col_ind] != data_matrix[row_ind, col_ind]
    data_matrix[row_ind, col_ind] = entry_value
time_end_loop = time.time()
print("Time used in the loop: ", time_end_loop - time_start_loop)
print("Minimum value: ", min_val)
data_matrix = data_matrix[:len(pair_list), :len(subgroup_list)]
# Now we have: na_counter: 4124417, inval_counter: 954536, goodval_counter: 66551.

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
#pdb.set_trace()

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
shuffled = np.random.permutation(range(len(pair_list)))
start_writing = time.time()
subgroup_list = ['toxcast_' + str(task_id) for task_id in subgroup_list]
with open('restructured_new.csv', 'w', newline='') as csvfile:
  fieldnames = subgroup_list + ['smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i in range(len(pair_list)):
    index = shuffled[i]
    pair = pair_list[index]
    data_line = ['' if np.isnan(entry) else entry for entry in data_matrix[index]]
    line_values = dict(zip(subgroup_list, data_line))
    out_line = {'smiles': pair[0], 'proteinName': pair[1], 'protein_dataset': 'toxcast'}
    line_values.update(out_line)
    writer.writerow(line_values)

end_writing = time.time()
print("Time spent in writing: ", end_writing - start_writing)