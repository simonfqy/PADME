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
from dcCustom.feat import Protein

def load_prot_dict(protein_list, prot_desc_path, sequence_field, 
  phospho_field):
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
    descriptor = row[2:]
    descriptor = np.array(descriptor)
    descriptor = np.reshape(descriptor, (1, len(descriptor)))
    pair = (source, row[0])      
    sequence = row[sequence_field]
    phosphorylated = row[phospho_field]
    protein = Protein(row[0], source, (phosphorylated, sequence))
    if protein not in set(protein_list):
      protein_list.append(protein)  
    

df = pd.read_csv('NCI60_bio.csv', header = 0, index_col=2)
#df = df.head(60000)
molList = list(df.index)
molList = [mol for mol in molList if mol==mol]
assert len(df) == len(molList)
selected_mol_set = set()
selected_mol_list = []
mols_to_choose = 1000
GIarray = np.asarray(df.iloc[:, 5])
sorted_indices = np.argsort(GIarray)
for i in sorted_indices:
  smiles = molList[i]
  if smiles not in selected_mol_set:
    selected_mol_set.add(smiles)
    selected_mol_list.append(smiles)
    if len(selected_mol_set) >= mols_to_choose:
      break

prot_desc_path_list = ['../davis_data/prot_desc.csv', '../metz_data/prot_desc.csv', 
  '../KIBA_data/prot_desc.csv', '../full_toxcast/prot_desc.csv']
prot_list = []

for path in prot_desc_path_list:
  load_prot_dict(prot_list, path, 1, 2)

start_writing = time.time()
with open('restructured.csv', 'w', newline='') as csvfile:
  fieldnames = ['smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for mol in selected_mol_list:
    for prot in prot_list:
      prot_source_and_name = prot.get_name()
      out_line = {'smiles': mol, 'proteinName': prot_source_and_name[1], 
        'protein_dataset': prot_source_and_name[0]}
      writer.writerow(out_line)

end_writing = time.time()
print("Time spent in writing: ", end_writing - start_writing)
