"""
Data Structures used to represented molecules for convolutions.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Qingyuan Feng"
__copyright__ = "Copyright 2018, Simon Fraser University"
__license__ = "MIT"
  
class Protein(object):
  """Holds only the names of the proteins, so that in the 
  ProtFeaturizer.featurize() function, the descriptors can 
  be fetched using those names from the corresponding file."""
  
  def __init__(self, name):
    self.name = name
    
  def __eq__(self, other):
    if isinstance(self, other.__class__):
      return self.name == other.name
    return False
    
  def __hash__(self):
    return hash(self.name)
    
  def get_name(self):
    return self.name

