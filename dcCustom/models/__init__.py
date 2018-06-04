"""
Gathers all models in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from dcCustom.models.tensorgraph.fcnet import MultiTaskRegressor
from dcCustom.models.tensorgraph.fcnet import MultiTaskClassifier
from dcCustom.models.models import Model
from dcCustom.models.tensorgraph.tensor_graph import TensorGraph
from dcCustom.models.tensorgraph.graph_models import WeaveTensorGraph, GraphConvTensorGraph, \
  MPNNTensorGraph
from dcCustom.models.tensorgraph.fcnet import MultiTaskRegressor, MultiTaskClassifier
