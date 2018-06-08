"""
Topological fingerprints.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Steven Kearnes, modified by Qingyuan Feng"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "LGPL v2.1+"

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from dcCustom.feat import Featurizer
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import pdb

class ComparableFingerprint(object):

  def __init__(self, mol, radius, nBits=2048, useChirality=False, useBondTypes=True, 
    useFeatures=False, smiles=None):
    self.fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
      mol, radius, nBits=nBits, useChirality=useChirality, useBondTypes=useBondTypes,
      useFeatures=useFeatures)
    self.fingerprint_array = np.asarray(self.fingerprint)
    self.smiles = smiles

  def __eq__(self, other):
    if isinstance(self, other.__class__):
      if self.smiles is not None and other.smiles is not None:
        return self.smiles == other.smiles
      else:
        return np.array_equal(self.fingerprint_array, other.fingerprint_array)
    return False

  def __hash__(self):
    assert self.smiles is not None
    return hash(self.smiles)

  def get_array(self):
    return self.fingerprint_array

class CircularFingerprint(Featurizer):
  """
  Circular (Morgan) fingerprints.

  Parameters
  ----------
  radius : int, optional (default 2)
      Fingerprint radius.
  size : int, optional (default 2048)
      Length of generated bit vector.
  chiral : bool, optional (default False)
      Whether to consider chirality in fingerprint generation.
  bonds : bool, optional (default True)
      Whether to consider bond order in fingerprint generation.
  features : bool, optional (default False)
      Whether to use feature information instead of atom information; see
      RDKit docs for more info.
  sparse : bool, optional (default False)
      Whether to return a dict for each molecule containing the sparse
      fingerprint.
  calc_smiles : bool, optional (default False)
      Whether to calculate SMILES strings for fragment IDs (only applicable
      when calculating sparse fingerprints).
  """
  name = 'circular'

  def __init__(self, radius=2, size=2048, chiral=False, bonds=True,
             features=False, sparse=False, calc_smiles=False):
    self.radius = radius
    self.size = size
    self.chiral = chiral
    self.bonds = bonds
    self.features = features
    self.sparse = sparse
    self.calc_smiles = calc_smiles

  def _featurize(self, mol, smiles=None):
    """
    Calculate circular fingerprint.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    if self.sparse:
      info = {}
      fp = rdMolDescriptors.GetMorganFingerprint(
          mol, self.radius, useChirality=self.chiral,
          useBondTypes=self.bonds, useFeatures=self.features,
          bitInfo=info)
      fp = fp.GetNonzeroElements()  # convert to a dict

      # generate SMILES for fragments
      if self.calc_smiles:
        fp_smiles = {}
        for fragment_id, count in fp.items():
          root, radius = info[fragment_id][0]
          env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root)
          frag = Chem.PathToSubmol(mol, env)
          smiles = Chem.MolToSmiles(frag)
          fp_smiles[fragment_id] = {'smiles': smiles, 'count': count}
        fp = fp_smiles
    else:
      fp = ComparableFingerprint(mol, self.radius, nBits=self.size, useChirality=self.chiral,
          useBondTypes=self.bonds, useFeatures=self.features, smiles=smiles)
    return fp
