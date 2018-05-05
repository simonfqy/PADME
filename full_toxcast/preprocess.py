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


df_uniprot = pd.read_csv('Assay_UniprotID_subgroup.csv', header = 0, index_col=0)
subgroup_dict = dict(df_uniprot.iloc[:]['Subgroup'])
protID_dict = dict(df_uniprot.iloc[:]['Uniprot_ID'])
protList = list(set(df_uniprot.iloc[:]['Uniprot_ID']))
subgroup_list = list(set(df_uniprot.iloc[:]['Subgroup']))
num_subgroups = len(subgroup_list)

df = pd.read_csv('Assays_smiles_INCHI.csv', header = 0, index_col=2)
assayList = list(df)[16:]
molList = list(df.index)
molList = [mol for mol in molList if mol==mol]
pair_dict = {}
smiles_indices = {}
invalid_mols = set()
duplicate_mols = set()
interactions = []
row_ind = 0
INACTIVE_VAL = 1000.0
min_val = 1000
max_value = 0
counter = 0
pair_list = []
row_index = 0
for j in range(len(df)):
  this_row = dict(df.iloc[j])
  assert df.index[j] == df.index[j]
  smiles = molList[j]
  for i, assay_name in enumerate(assayList):
    prot_id = protID_dict[assay_name]
    group_id = subgroup_dict[assay_name]
    pair = (smiles, prot_id)
    if pair not in pair_list:
      pair_list.append(pair)
    entry_value = this_row[assay_name]
    if entry_value == 1000000:
      entry_value = INACTIVE_VAL

    row_index += 1


for row in df.itertuples():
  if row[0] != row[0]:
    continue
  smiles = row[0]
  values = list(row[17:])
  if smiles not in smiles_indices:
    smiles_indices[smiles] = [row_ind]
  else:
    smiles_indices[smiles].append(row_ind)
  for i, element in enumerate(values):

    if element != element:
      continue
    matchObj = re.match(r'-?\d', str(element))
    if not matchObj:
      values[i] = np.nan
      continue
    value = float(element)
    if value == 1000000:
      value = INACTIVE_VAL
      values[i] = value
      counter += 1
    elif value > max_value:
      max_value = value
    if value < min_val and value != 0:
      min_val = value
    assay = assayList[i]
    # pair = (smiles, prot)
    # if pair not in pair_dict:
    #   pair_dict[pair] = value
    # else:
    #   duplicate_mols.add(smiles)
    #   if pair_dict[pair] != value:
    #     invalid_mols.add(smiles)

  interactions.append(values)
  row_ind += 1
#print(min_val)
#pdb.set_trace()
interactions = np.array(interactions)
interactions = 4 - np.log10(interactions)
max_val = 4 - np.log10(min_val)
inf_indices = np.where(np.isinf(interactions))
num_inf_entries = len(inf_indices[0])
for i in range(num_inf_entries):
  first_ind = inf_indices[0][i]
  second_ind = inf_indices[1][i]
  assert np.isinf(interactions[first_ind, second_ind])
  interactions[first_ind, second_ind] = np.ceil(max_val)

num_bins = 20
fig, ax = plt.subplots()
#n, bins, patches = ax.hist(interactions, num_bins, density=1)
interaction_flattened = interactions.flatten()
interaction_flattened = interaction_flattened[~np.isnan(interaction_flattened)]
ax.hist(interaction_flattened, num_bins)
#ax.hist(interaction_flattened, num_bins, range=(1.01, np.ceil(max_val)))
ax.set_xlabel('ToxCast assay values')
ax.set_ylabel('Occurrence')
ax.set_title('Histogram of ToxCast assay values')
#ax.plot(bins)
#fig.tight_layout()
#plt.show()
#pdb.set_trace()

counter = 0
dup_indices = {smiles: inds for (smiles, inds) in smiles_indices.items()
  if smiles in (duplicate_mols - invalid_mols)}
with open('restructured.csv', 'w', newline='') as csvfile:
  fieldnames = ['interaction_value', 'smiles', 'proteinName']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i, compound in enumerate(molList):
    if compound in invalid_mols:
      continue
    for j, protein in enumerate(protList):
      intxn_value = interactions[i][j]
          
      if intxn_value != intxn_value:        
        continue
      # if not np.isfinite(intxn_value):
      #   intxn_value = np.ceil(max_val)
      #   counter += 1 
      writer.writerow({'interaction_value': intxn_value,
        'smiles': compound, 'proteinName': protein})
  #pdb.set_trace()
print("counter: ", counter)
