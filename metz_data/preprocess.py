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

df = pd.read_csv('Metz_interaction.csv', header = 0, index_col=0, usecols=range(2, 186))
#df = df.head(3)
protList = list(df)[11:]
molList = list(df.index)
molList = [molName for molName in molList if molName == molName]
prot_dict = {i: prot for (i, prot) in enumerate(protList)}
pair_dict = {}
smiles_indices = {}
invalid_mols = set()
duplicate_mols = set()
#print(len(molList))
interactions = []
row_ind = 0
for row in df.itertuples():  
  values = list(row)
  if values[0] != values[0]:
    continue
  smiles = values[0]
  values = values[12:]
  intxn = []
  if smiles not in smiles_indices:
    smiles_indices[smiles] = [row_ind]
  else:
    smiles_indices[smiles].append(row_ind)
  #pdb.set_trace()  
  for i, element in enumerate(values):
    if element == element: #Not a NAN value
      matchObj = re.match('\d', element)
      if not matchObj:
        value = np.nan
      else:
        value = float(element)
        prot = prot_dict[i]        
        pair = (smiles, prot)
        if pair not in pair_dict:
          pair_dict[pair] = value
        else:
          duplicate_mols.add(smiles)
          if pair_dict[pair] != value:            
            invalid_mols.add(smiles) 
    else:
      value = np.nan
    intxn.append(value)  
  interactions.append(intxn)
  row_ind += 1

#interactions = np.array(interactions)
#interaction_bin = (interactions >= 7.6) * 1
counter = 0
dup_indices = {smiles: inds for (smiles, inds) in smiles_indices.items()
  if smiles in (duplicate_mols - invalid_mols)}
'''
invalid_indices = {smiles: inds for (smiles, inds) in smiles_indices.items()
  if smiles in invalid_mols}

count = 0
pair_values = {}
for smiles, indices in invalid_indices.items():
  for k, protein in prot_dict.items():
    discrepancy = False
    for m in range(len(indices)):
      value = interactions[indices[m]][k]
      if value != value:
        continue
      pair = (smiles, protein)
      if pair not in pair_values:
        pair_values[pair] = value
        continue
      if value != pair_values[pair]:
        discrepancy = True
        count += 1
    if discrepancy:
      # Don't need to change indices[1] beyond, because they will not be examined
      # in the writing process.
      interactions[indices[0]][k] = np.nan
'''

pdb.set_trace()
# Stores the duplicate molecules which have been processed.
processed_duplicate_mols_ind = set()

with open('restructured3.csv', 'w', newline='') as csvfile:
  fieldnames = ['metz', 'smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i, compound in enumerate(molList):
    if compound in invalid_mols:
      continue
    mol_inds = [i]
    pair_set = set()
    if compound in duplicate_mols:
      mol_inds = dup_indices[compound]
    for j in mol_inds:
      if j in processed_duplicate_mols_ind:
        continue      
      for k, protein in prot_dict.items():        
        intxn_value = interactions[j][k]
        #intxn_bin = interaction_bin[i][j]
           
        if intxn_value != intxn_value:
          continue
        if len(mol_inds) > 1:
          pair = (compound, protein)
          if pair in pair_set:
            counter += 1
            continue
          pair_set.add(pair)     
        writer.writerow({'metz': intxn_value, 'smiles': compound, 
          'proteinName': protein, 'protein_dataset': 'metz'})
      if len(mol_inds) > 1:
        processed_duplicate_mols_ind.add(j)
    
