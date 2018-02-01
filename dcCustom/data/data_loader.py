"""
Process an input dataset into a format suitable for machine learning.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gzip
import pandas as pd
import numpy as np
import csv
import numbers
import tempfile
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit import Chem
import time
import sys
from deepchem.utils.save import log
from deepchem.utils.save import load_csv_files
from deepchem.utils.save import load_sdf_files
from deepchem.utils.save import encode_fasta_sequence
from deepchem.feat import UserDefinedFeaturizer
from deepchem.data import DiskDataset
