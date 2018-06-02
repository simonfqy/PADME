from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import sys
import pdb
import csv

def get_cid():
  
  chembl_fname = "Chembl_ids.txt"
  with open(chembl_fname, 'r') as f:
    data = f.readlines()
    for line in data:
      line = line.split()[0]    
      chembl_id_list.append(line)

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

if __name__ == '__main__':
  get_cid()