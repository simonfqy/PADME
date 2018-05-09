"""
Making it easy to import in classes.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

#from dcCustom.feat.base_classes import ProtFeaturizer
from dcCustom.feat.base_classes import Featurizer
from dcCustom.feat.proteins import Protein
from dcCustom.feat.graph_features import WeaveFeaturizer
from dcCustom.feat.graph_features import ConvMolFeaturizer
from dcCustom.feat.fingerprints import CircularFingerprint
from dcCustom.feat.fingerprints import ComparableFingerprint