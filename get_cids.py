from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import os
import time
import sys
import pwd
import pdb
import csv

chembl_id_list = []
with open("Chembl_ids.txt", 'r') as f:
  data = f.readlines()
  for line in data:
    line = line.split()[0]    
    chembl_id_list.append(line)
#pdb.set_trace()
chembl_to_cid = {}
with open("chembl_to_cids.txt", 'r') as f:
  data = f.readlines()
  for line in data:
    words = line.split()
    if words[0] not in chembl_to_cid:
      chembl_to_cid[words[0]] = words[1]
    else:
      assert False

with open("compound_cids.txt", 'w') as f:
  cid_list = []
  for chembl_id in chembl_id_list:
    cid = chembl_to_cid[chembl_id]
    cid_list.append(cid)
  f.write('\n'.join(cid_list))