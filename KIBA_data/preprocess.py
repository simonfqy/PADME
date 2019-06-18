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
from collections import OrderedDict


def generate_data(input_csv, binarize=False, head_only=False, head_row_num=15000, 
  limit_rows=False, limit_row_num=2400, prefix="kiba_", input_prot=True, output_csv=None):

  df = pd.read_csv(input_csv, header = 0, index_col=1)
  if head_only:
    df = df.head(head_row_num)
  protList = list(df)[1:]
  molList = list(df.index)
  molList = [mol for mol in molList if mol==mol]
  pair_dict = {}
  smiles_to_indices = {}
  invalid_mols = set()
  duplicate_mols = set()
  interactions = []
  row_ind = 0
  for row in df.itertuples():
    if row[0] != row[0]:
      continue
    smiles = row[0]
    values = list(row[2:])
    if smiles not in smiles_to_indices:
      smiles_to_indices[smiles] = [row_ind]
    else:
      smiles_to_indices[smiles].append(row_ind)
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

  if binarize:
    interactions = np.array(interactions)
    interaction_bin = (interactions <= 3.0) * 1

  # The two sets, invalid_mols and duplicate_mols are the same.
  
  with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['smiles']
    if input_prot:
      fieldnames = ['kiba'] + fieldnames + ['proteinName', 'protein_dataset']
      if binarize:
        fieldnames = ['kiba_bin'] + fieldnames
    else:
      tasks = [prefix + prot for prot in protList]
      fieldnames = tasks + fieldnames
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i, compound in enumerate(molList):
      if compound in invalid_mols:
        continue
      output_dict = {'smiles': compound}
      if input_prot:
        for j, protein in enumerate(protList):
          intxn_value = interactions[i][j]
              
          if intxn_value != intxn_value:        
            continue
          output_dict.update({'kiba': intxn_value, 'proteinName': protein, 'protein_dataset': 'kiba'})
          writer.writerow(output_dict)
      else:
        for j, protein in enumerate(protList):
          intxn_value = interactions[i][j]
          if intxn_value != intxn_value:        
            intxn_value = ''
          task_name = fieldnames[j]
          output_dict[task_name] = intxn_value
        writer.writerow(output_dict)
        

def filter_data(input_file, filter_threshold=1, output_csv=None):
  df = pd.read_csv(input_file, header = 0, index_col=False)
  headers = list(df)  
  finished = False
  while not finished:
    # Each row of the df corresponds to a molecule, each column corresponds to a protein.
    is_not_null_df = df.notnull()
    # Sum the columns first.
    col_sum_nonnull_entries = is_not_null_df.sum()
    deleted_column_names = []
    if any(col_sum_nonnull_entries <= filter_threshold):
      col_name_to_nonnull_num = OrderedDict(col_sum_nonnull_entries)
      for col_name, nonnull_num in col_name_to_nonnull_num.items():
        if nonnull_num > filter_threshold:
          continue
        deleted_column_names.append(col_name)
    df = df.drop(deleted_column_names, axis=1)
    is_not_null_df = is_not_null_df.drop(deleted_column_names, axis=1)
    print("deleted column number: ", len(deleted_column_names))

    # Then sum the rows.
    row_sum_nonnull_entries = is_not_null_df.sum(axis=1)
    deleted_row_inds = []
    if any(row_sum_nonnull_entries <= filter_threshold + 1):
      row_ind_to_nonnull_num = OrderedDict(row_sum_nonnull_entries)
      for row_ind, nonnull_num in row_ind_to_nonnull_num.items():
        if nonnull_num > filter_threshold + 1:
          continue
        deleted_row_inds.append(row_ind)
    df = df.drop(deleted_row_inds)
    is_not_null_df = is_not_null_df.drop(deleted_row_inds)
    print("deleted row number: ", len(deleted_row_inds))

    col_sum_nonnull_entries = is_not_null_df.sum()
    if all(col_sum_nonnull_entries > filter_threshold):
      finished = True

  # Output.
  df.to_csv(output_csv, index=False)


if __name__ == '__main__':
  # generate_data('Smiles_bio_results.csv', input_prot=False, output_csv='restructured_no_prot_unfiltered.csv')
  filter_data('restructured_no_prot_unfiltered.csv', filter_threshold=6, output_csv='restructured_no_prot.csv')