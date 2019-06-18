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
  limit_rows=False, limit_row_num=2400, prefix="metz_", input_prot=True, output_csv=None):

  df = pd.read_csv(input_csv, header = 0, index_col=0, usecols=range(2, 186))
  if head_only:
    df = df.head(head_row_num)
  protList = list(df)[11:]
  molList = list(df.index)
  molList = [molName for molName in molList if molName == molName]
  prot_dict = {i: prot for (i, prot) in enumerate(protList)}
  pair_dict = {}
  smiles_to_indices = {}
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
    if smiles not in smiles_to_indices:
      smiles_to_indices[smiles] = [row_ind]
    else:
      smiles_to_indices[smiles].append(row_ind)
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

  if binarize:
    interactions = np.array(interactions)
    interaction_bin = (interactions >= 7.6) * 1

  counter = 0
  dup_indices = {smiles: inds for (smiles, inds) in smiles_to_indices.items()
    if smiles in (duplicate_mols - invalid_mols)}

  # Stores the duplicate molecules which have been processed.
  processed_duplicate_mols = set()

  with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['smiles']
    if input_prot:
      fieldnames = ['metz'] + fieldnames + ['proteinName', 'protein_dataset']
      if binarize:
        fieldnames = ['metz_bin'] + fieldnames
    else:
      tasks = [prefix + prot for prot in protList]
      fieldnames = tasks + fieldnames

    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    
    for i, compound in enumerate(molList):
      if compound in invalid_mols:
        continue
      if compound in processed_duplicate_mols:
        continue
      output_dict = {'smiles': compound}
      mol_inds = [i]
      pair_set = set()
      if compound in duplicate_mols:
        mol_inds = dup_indices[compound]
        processed_duplicate_mols.add(compound)

      if input_prot: 
        for j in mol_inds:      
          for k, protein in prot_dict.items():        
            intxn_value = interactions[j][k]
                      
            if intxn_value != intxn_value:
              continue
            if len(mol_inds) > 1:
              pair = (compound, protein)
              if pair in pair_set:
                counter += 1
                continue
              pair_set.add(pair) 
            output_dict.update({'metz': intxn_value, 'proteinName': protein, 'protein_dataset': 'metz'})    
            writer.writerow(output_dict)

      else:
        for j in mol_inds:          
          for k, protein in prot_dict.items():
            intxn_value = interactions[j][k]
            if intxn_value != intxn_value:
              intxn_value = ''             
            task_name = fieldnames[k]
            if len(mol_inds) > 1:
              pair = (compound, protein)
              if pair in pair_set:
                counter += 1 
                if intxn_value == '':
                  continue 
                if output_dict[task_name] != '':
                  assert output_dict[task_name] == intxn_value                
                         
              pair_set.add(pair)
            output_dict[task_name] = intxn_value

        writer.writerow(output_dict)   
    print("counter: ", str(counter))   


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
  # generate_data('Metz_interaction.csv', input_prot=False, output_csv='restructured_no_prot_unfiltered.csv')
  filter_data('restructured_no_prot_unfiltered.csv', filter_threshold=1, output_csv='restructured_no_prot.csv')