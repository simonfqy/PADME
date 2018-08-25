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

df = pd.read_csv('restructured_unique_old.csv', header = 0, index_col=False, usecols=range(3))

with open('restructured_unique.csv', 'w', newline='') as csvfile:
  fieldnames = ['interaction_value', 'smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i in range(len(df)):    
    writer.writerow({'interaction_value': df.iloc[i, 0],
      'smiles': df.iloc[i, 1], 'proteinName': df.iloc[i, 2], 'protein_dataset': 'kiba'})
      
