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
invalid_val = -1.0
#print(len(molList))
interactions = []
rowCount = 1 
for row in df.itertuples():  
  values = list(row)
  if values[0] != values[0]:
    continue
  values = values[12:]
  intxn = []  
  for element in values:
    if element == element: #Not a NAN value
      matchObj = re.match('\d', element)
      if not matchObj:
        intxn.append(invalid_val)
      else:
        intxn.append(float(element))
    else:
      intxn.append(invalid_val)  
  interactions.append(intxn)
  rowCount += 1  
  #print(interactions)
interactions = np.array(interactions)
#interaction_bin = (interactions >= 7.6) * 1
counter = 0
#pdb.set_trace()
with open('restructured.csv', 'w', newline='') as csvfile:
  fieldnames = ['interaction_value', 'smiles', 'proteinName']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i, compound in enumerate(molList):
    for j, protein in enumerate(protList):
      # will start writing rows.
      intxn_value = interactions[i][j]
      #intxn_bin = interaction_bin[i][j]      
      if intxn_value == -1:        
        continue
      writer.writerow({'interaction_value': intxn_value,
        'smiles': compound, 'proteinName': protein})
      #counter += 1
  #     if (counter > 2400):
  #       break
  # if (counter > 2400):
  #   break

            
