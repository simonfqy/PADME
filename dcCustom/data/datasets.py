"""
Contains wrapper class for datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import math
import numpy as np
import pandas as pd
import random
from deepchem.utils.save import save_to_disk, save_metadata
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import log
from pandas import read_hdf
import tempfile
import time
import shutil
import json
from multiprocessing.dummy import Pool

__author__ = "Bharath Ramsundar, modified by Qingyuan Feng"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"