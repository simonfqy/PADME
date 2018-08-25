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

df_mols = pd.read_csv('mol.csv', header = 0, index_col=False)
mol_mapping = {}
for i in range(len(df_mols)):
  mol_mapping[df_mols.iloc[i][0]] = df_mols.iloc[i][1]
df_old_df = pd.read_csv('restructured_old.csv', header = 0, index_col=False)
subgroup_list = list(df_old_df)[:-3]
#pdb.set_trace()
time1 = time.time()
with open('restructured_new.csv', 'w', newline='') as csvfile:
  fieldnames = subgroup_list + ['smiles', 'proteinName', 'protein_dataset']
  writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
  writer.writeheader()
  for i in range(len(df_old_df)):
    index = i
    data_line = ['' if np.isnan(entry) else entry for entry in df_old_df.iloc[index][:-3]]
    #data_line = df_old_df.iloc[index][:-3]
    line_values = dict(zip(subgroup_list, data_line))
    molecule = df_old_df.iloc[index][-3]
    if molecule in mol_mapping:
      molecule = mol_mapping[molecule]
    out_line = {'smiles': molecule, 'proteinName': df_old_df.iloc[index][-2],
     'protein_dataset': df_old_df.iloc[index][-1]}
    line_values.update(out_line)
    writer.writerow(line_values)
time2 = time.time()
print(time2-time1)