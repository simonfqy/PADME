from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import argparse
import os
import sys
import pdb
import csv


def generate_data(input_csv, binarize=False, head_only=False, head_row_num=15000, 
    limit_rows=False, limit_row_num=2400, prefix="davis_", input_prot=True, output_csv=None):
    df = pd.read_csv(input_csv, header = 2, index_col=0, usecols=range(3, 76))
    if head_only:
        df = df.head(head_row_num)
    molList = list(df)
    #print(len(molList))
    protList = list(df.index)
    interactions = [] 
    for row in df.itertuples():
        intxn = list(row)[1:]
        interactions.append(intxn)  
        #print(interactions)
    interactions = np.array(interactions)
    interactions[np.isnan(interactions)] = 10000
    interactions = 9 - np.log10(interactions)
    if binarize:
        interaction_bin = (interactions >= 7.0) * 1
    if limit_rows:
        counter = 0
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['smiles']
        if input_prot:
            fieldnames = ['davis'] + fieldnames + ['proteinName', 'protein_dataset']
            if binarize:
                fieldnames = ['davis_bin'] + fieldnames
        else:
            tasks = [prefix + prot for prot in protList]
            fieldnames = tasks + fieldnames
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()

        if input_prot:
            for i, protein in enumerate(protList):     
                output_dict = {'proteinName': protein, 'protein_dataset': 'davis'}
                for j, compound in enumerate(molList):
                    # will start writing rows.
                    intxn_value = interactions[i][j]
                    output_dict.update({'davis': intxn_value, 'smiles': compound})
                    if binarize:
                        intxn_bin = interaction_bin[i][j]
                        output_dict['davis_bin'] = intxn_bin 
                    writer.writerow(output_dict)
                    if not limit_rows:
                        continue
                    counter += 1
                    if (counter > limit_row_num):
                        break
                if not limit_rows:
                    continue
                if (counter > limit_row_num):
                    break
        else:
            for j, compound in enumerate(molList):
                output_dict = {'smiles': compound}
                for i, _ in enumerate(protList):
                    task_name = fieldnames[i]
                    output_dict[task_name] = interactions[i][j]
                writer.writerow(output_dict)

if __name__ == '__main__':
    generate_data('Bio_results.csv', input_prot=True, limit_rows=True, limit_row_num=2400, 
        output_csv='restructured_toy.csv')
            
