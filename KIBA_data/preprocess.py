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

df = pd.read_csv('Smiles_bio_results.csv', header = 0, index_col=1)
protList = list(df)[1:]
molList = list(df.index)
molList = [mol for mol in molList if mol==mol]
pair_dict = {}
smiles_indices = {}
invalid_mols = set()
duplicate_mols = set()
interactions = []
row_ind = 0
for row in df.itertuples():
  if row[0] != row[0]:
    continue
  smiles = row[0]
  values = list(row[2:])
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
    prot = protList[i]
    pair = (smiles, prot)
    if pair not in pair_dict:
      pair_dict[pair] = value
    else:
      duplicate_mols.add(smiles)
      if pair_dict[pair] != value:
        invalid_mols.add(smiles)

  interactions.append(values)
  row_ind += 1

dup_indices = {smiles: inds for (smiles, inds) in smiles_indices.items()
  if smiles in (duplicate_mols - invalid_mols)}
#pdb.set_trace()
with open('restructured2.csv', 'w', newline='') as csvfile:
  fieldnames = ['kiba', 'smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i, compound in enumerate(molList):
    if compound in invalid_mols:
      continue
    for j, protein in enumerate(protList):
      intxn_value = interactions[i][j]
          
      if intxn_value != intxn_value:        
        continue
      writer.writerow({'kiba': intxn_value, 'smiles': compound, 'proteinName': protein,
       'protein_dataset': 'kiba'})
      #counter += 1