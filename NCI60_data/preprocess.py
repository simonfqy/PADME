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
import dcCustom
from dcCustom.feat import Protein

invalid_mols = ['O(C)C1(C)C(OC)C(OC2c3c([O-])c4C(=O)c5c([O-])cc6C7(C)C(O)C([N+H](C)C)C(O)C(O7)Oc6c5C(=O)c4cc3CC(O)(C)C2)OC(C)C1OC',
  "O=C(OC1C(C)C(O)C(C)CCC(OC)/C(/C)=C\CC(C)CC(OC)/C=C/CC(C(C(O)C(CCC(OC(=O)C([N+H](C)C)C)C(C(OC(=O)C)C(/C=C/N(C=O)C)C)C)C)C)OC(=O)/C=C/C=C/C1)C([N+H](C)C)COC",
  "O=C(OCC)Nc1[n+H]c(N)c2N=C([C@H](C)Nc2c1)c1ccccc1",
  "FCCOC(=O)Nc1[n+H]c(N)c2N=C([C@H](C)Nc2c1)c1ccccc1"]

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
    
def produce_dataset():
  df = pd.read_csv('NCI60_bio.csv', header = 0, index_col=2)
  df = df.head(60000)
  molList = list(df.index)
  molList = [mol for mol in molList if mol==mol]
  assert len(df) == len(molList)
  selected_mol_set = set()
  selected_mol_list = []
  mols_to_choose = 18
  GIarray = np.asarray(df.iloc[:, 5])
  sorted_indices = np.argsort(GIarray)
  for i in sorted_indices:
    smiles = molList[i]
    if smiles not in selected_mol_set:
      selected_mol_set.add(smiles)
      selected_mol_list.append(smiles)
      if len(selected_mol_set) >= mols_to_choose:
        break

  tasks, _, _ = dcCustom.molnet.load_toxcast(featurizer="ECFP", currdir="../")

  prot_desc_path_list = ['../davis_data/prot_desc.csv', '../metz_data/prot_desc.csv', 
    '../KIBA_data/prot_desc.csv', '../full_toxcast/prot_desc.csv']
  prot_list = []

  for path in prot_desc_path_list:
    load_prot_dict(prot_list, path, 1, 2)

  start_writing = time.time()
  with open('restructured_toy.csv', 'w', newline='') as csvfile:
    fieldnames = tasks + ['smiles', 'proteinName', 'protein_dataset']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for mol in selected_mol_list:
      if mol in invalid_mols:
        continue
      for prot in prot_list:
        prot_source_and_name = prot.get_name()
        out_line = {'smiles': mol, 'proteinName': prot_source_and_name[1], 
          'protein_dataset': prot_source_and_name[0]}
        line_values = dict(zip(tasks, [0]*len(tasks)))
        out_line.update(line_values)
        writer.writerow(out_line)

  end_writing = time.time()
  print("Time spent in writing: ", end_writing - start_writing)

def synthesize_ranking(prediction_file, output_file):
  df = pd.read_csv('../full_toxcast/restructured.csv', header=0, index_col=False)
  tasks, _, _ = dcCustom.molnet.load_toxcast(featurizer="ECFP", currdir="../")
  datapoints = []
  for task in tasks:
    task_vector = np.asarray(df.loc[:, task])
    datapoints.append(np.count_nonzero(~np.isnan(task_vector)))
  datapoints = np.array(datapoints)
  fractions = []
  for i in range(len(datapoints)):
    fractions.append(datapoints[i]/datapoints.sum())

  preds_df = pd.read_csv(prediction_file, header=0, index_col=False)
  compounds = preds_df.loc[:, 'Compound']
  prot_names = preds_df.loc[:, 'proteinName']
  prot_sources = preds_df.loc[:, 'protein_dataset']
  composite_preds = np.zeros_like(preds_df.loc[:, tasks[0]])
  for j in range(len(tasks)):
    task = tasks[j]
    pred_task = preds_df.loc[:, task] * fractions[j]
    composite_preds += pred_task
  neg_composite_preds = -1 * composite_preds
  sorted_indices = neg_composite_preds.argsort()
  with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['smiles', 'proteinName', 'protein_dataset', 'synthesized_score']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i in sorted_indices:
      out_line = {'smiles': compounds[i], 'proteinName': prot_names[i], 
        'protein_dataset': prot_sources[i], 'synthesized_score': composite_preds[i]}
      writer.writerow(out_line)  


if __name__ == "__main__":
  synthesize_ranking('preds_tc_ecfp.csv', 'synthesized_values.csv')