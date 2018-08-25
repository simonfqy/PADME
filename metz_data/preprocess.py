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

df = pd.read_csv('Bio_results.csv', header = 2, index_col=0, usecols=range(3, 76))
#df = df.head(15000)
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
interaction_bin = (interactions >= 7.0) * 1
counter = 0
with open('restructured_bin2.csv', 'w', newline='') as csvfile:
    fieldnames = ['davis_bin', 'davis', 'smiles', 'proteinName']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for i, protein in enumerate(protList):
        for j, compound in enumerate(molList):
            # will start writing rows.
            intxn_value = interactions[i][j]
            intxn_bin = interaction_bin[i][j]
            #if  np.isnan(intxn_value):
            #    intxn_value = 10000
            writer.writerow({'davis_bin': intxn_bin, 'davis': intxn_value,
              'smiles': compound, 'proteinName': protein})
            counter += 1
            if (counter > 2400):
                break
        if (counter > 2400):
            break

            
