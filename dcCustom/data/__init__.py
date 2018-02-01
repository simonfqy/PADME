"""
Gathers all datasets in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#from deepchem.data.datasets import pad_features
#from deepchem.data.datasets import pad_batch
from dcCustom.data.datasets import Dataset
from dcCustom.data.datasets import NumpyDataset
from dcCustom.data.datasets import DiskDataset
#from deepchem.data.datasets import sparsify_features
#from deepchem.data.datasets import densify_features
#from deepchem.data.supports import *
from dcCustom.data.data_loader import DataLoader
from dcCustom.data.data_loader import CSVLoader
