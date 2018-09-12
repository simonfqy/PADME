"""
Data Structures used to represented molecules for convolutions.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import re

__author__ = "Qingyuan Feng"
__copyright__ = "Copyright 2018, Simon Fraser University"
__license__ = "MIT"
  
class Protein(object):
  """Holds only the names of the proteins, so that in the 
  ProtFeaturizer.featurize() function, the descriptors can 
  be fetched using those names from the corresponding file."""
  
  def __init__(self, name, source, sequence):
    self.name = name
    self.source = source
    self.sequence = sequence
    
  def __eq__(self, other):
    if isinstance(self, other.__class__):
      if self.sequence == other.sequence:
        if self.source == other.source:
          return (self.name == other.name)
        # We need to tackle the problem brought by phosphorylation in Davis data.
        self_phospho_match = re.search('-phospho', self.name, re.I)
        other_phospho_match = re.search('-phospho', other.name, re.I)
        if self_phospho_match and other_phospho_match:
          return True
        # Here we made the assumption that those with '-nonphosphorylated' and those without
        # any descriptions of phosphorylation are the same if they have the same sequence.
        if not (self_phospho_match or other_phospho_match):
          return True
        return False
      else:
        return False
    return False
    
  def __hash__(self):
    assert self.sequence is not None
    return hash(self.sequence)
    
  def get_name(self):
    return (self.source, self.name)

  def get_sequence(self):
    return self.sequence

