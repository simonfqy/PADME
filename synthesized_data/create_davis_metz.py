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
import time

def load_sequence_dict(sequence_df, sequence_field, phospho_field):
  prot_desc_dict = {}
  for row in sequence_df.itertuples():
    sequence = row[sequence_field]
    phosphorylated = row[phospho_field]
    prot_desc_dict[row[0]] = [phosphorylated, sequence]
  return prot_desc_dict

'''
dfdavis = pd.read_csv('../davis_data/Bio_results.csv', header = 2, index_col=0, usecols=range(3, 76))
molListdavis = list(dfdavis)
dfmetz = pd.read_csv('../metz_data/Metz_interaction.csv', header = 0, index_col=0, usecols=range(2, 16))
molListmetz = list(dfmetz.index)
molListmetz = [molName for molName in molListmetz if molName == molName]
davis_mol = set(molListdavis)
metz_mol = set(molListmetz)
pdb.set_trace()
'''
dfm = pd.read_csv('../metz_data/restructured.csv', header = 0, index_col=False)
dfd = pd.read_csv('../davis_data/restructured.csv', header = 0, index_col=False)
#dfm = dfm.head(10)
#dfd = dfd.head(200)
#dfm = dfm.sort_values(by=['proteinName', 'smiles'])
#dfd = dfd.sort_values(by=['proteinName', 'smiles'])

metz_sequence_df = pd.read_csv('../metz_data/prot_desc_Metz.csv', header=0, 
  index_col=0, usecols=range(0, 6))
davis_sequence_df = pd.read_csv('../davis_data/prot_desc.csv', header=0, 
  index_col=0, usecols=range(0, 3))
time_start = time.time()
metz_sequence_dict = load_sequence_dict(metz_sequence_df, 4, 5)
davis_sequence_dict = load_sequence_dict(davis_sequence_df, 1, 2)
#pdb.set_trace()

invalid_val = -100
davis_value = []
metz_value = []
protein_name = []
protein_origin = []
cmpd_smiles = []
davis_leftout = [True]*len(dfd)
prot_correspondence_dict = {}
cmpd_correspondence_dict = {}

for i in range(len(dfm)):
  metz_prot = dfm.iloc[i]['proteinName']
  metz_cmpd = dfm.iloc[i]['smiles']
  metz_intxn = dfm.iloc[i]['interaction_value']
  prot_identity = metz_sequence_dict[metz_prot]
  if metz_prot not in prot_correspondence_dict:
    prot_correspondence = [davis_sequence_dict[a] == prot_identity for a 
      in dfd.iloc[:]['proteinName']]
    prot_correspondence_dict[metz_prot] = prot_correspondence
  else:
    prot_correspondence = prot_correspondence_dict[metz_prot]

  if metz_cmpd not in cmpd_correspondence_dict:
    cmpd_correspondence = [metz_cmpd == b for b in dfd.iloc[:]['smiles']]
    cmpd_correspondence_dict[metz_cmpd] = cmpd_correspondence
  else:
    cmpd_correspondence = cmpd_correspondence_dict[metz_cmpd]
  
  exact_correspondence = [a and b for a, b in zip(prot_correspondence, 
    cmpd_correspondence)]
  #if sum(prot_correspondence) > 0:
  #  pdb.set_trace()
  if sum(exact_correspondence):
    # There are exact correspondences.
    position = np.where(exact_correspondence)[0]
    assert len(position) == 1
    for pos in position:
      davis_intxn = dfd.iloc[pos]['interaction_value']
      davis_leftout[pos] = False
  else:
    davis_intxn = invalid_val
  davis_value.append(davis_intxn)
  metz_value.append(metz_intxn)
  protein_name.append(metz_prot)
  protein_origin.append('metz')
  cmpd_smiles.append(metz_cmpd)

#pdb.set_trace()
remaining_davis_entries = np.where(davis_leftout)[0]
for entry in remaining_davis_entries:
  davis_prot = dfd.iloc[entry]['proteinName']
  davis_cmpd = dfd.iloc[entry]['smiles']
  prot_identity = davis_sequence_dict[davis_prot]
  davis_value.append(dfd.iloc[entry]['interaction_value'])
  metz_value.append(invalid_val)
  protein_name.append(davis_prot)
  protein_origin.append('davis')
  cmpd_smiles.append(davis_cmpd)
  davis_leftout[entry] = False

metz_value = np.array(metz_value)
davis_value = np.array(davis_value)
#interaction_bin = (interactions >= 7.6) * 1

shuffled = np.random.permutation(range(len(metz_value)))
with open('davis-metz.csv', 'w', newline='') as csvfile:
  fieldnames = ['davis', 'metz', 'smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i in range(len(davis_value)):    
    # will start writing rows.    
    #intxn_bin = interaction_bin[i][j]
    index = shuffled[i]
    davis_val = davis_value[index]    
    metz_val = metz_value[index]      
           
    davis_val = '' if davis_val == invalid_val else davis_val    
    metz_val = '' if metz_val == invalid_val else metz_val
    writer.writerow({'davis': davis_val, 'metz': metz_val,
      'smiles': cmpd_smiles[index], 'proteinName': protein_name[index],
      'protein_dataset': protein_origin[index]})

time_finish = time.time()
print("time used: ", time_finish-time_start)