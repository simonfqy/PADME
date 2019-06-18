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
import pdb
from deepchem.utils.save import log
from deepchem.utils.save import load_csv_files
#from deepchem.utils.save import load_sdf_files
#from deepchem.utils.save import encode_fasta_sequence
from deepchem.feat import UserDefinedFeaturizer
from dcCustom.data import DiskDataset
from dcCustom.feat import Protein


def convert_df_to_numpy(df, tasks, verbose=False):
  """Transforms a dataframe containing deepchem input into numpy arrays"""
  n_samples = df.shape[0]
  n_tasks = len(tasks)

  time1 = time.time()
  y = np.hstack(
      [np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])
  time2 = time.time()

  w = np.ones((n_samples, n_tasks))
  missing = np.zeros_like(y).astype(int)
  feature_shape = None

  for ind in range(n_samples):
    for task in range(n_tasks):
      if y[ind, task] == "":
        missing[ind, task] = 1

  # ids = df[id_field].values
  # Set missing data to have weight zero
  for ind in range(n_samples):
    for task in range(n_tasks):
      if missing[ind, task]:
        y[ind, task] = 0.
        w[ind, task] = 0.

  return y.astype(float), w.astype(float)
  
def featurize_protein(df, field, source_field, prot_seq_dict, log_every_N=500, verbose=True):
  '''This is supposed to match the format of functions for featurizing molecules.
  It is not really featurizing, but only constructs the protein objects from their names.'''
  elems = df[field].tolist()
  sources = df[source_field].tolist()
  proteins = []
  for ind, prot in enumerate(elems):
    source = sources[ind]
    pair = (source, prot)
    sequence = prot_seq_dict[pair]
    proteins.append([Protein(prot, source = source, sequence = sequence)])  
  #return np.squeeze(np.array(proteins), axis=1), valid_inds
  return np.array(proteins)
 
def featurize_smiles_df(df, featurizer, field, log_every_N=1000, verbose=True):
  """Featurize individual compounds in dataframe.

  Given a featurizer that operates on individual chemical compounds 
  or macromolecules, compute & add features for that compound to the 
  features dataframe
  """
  sample_elems = df[field].tolist()

  features = []
  stderr_fileno = sys.stderr.fileno()
  stderr_save = os.dup(stderr_fileno)
  stderr_fd = open('./logs/error.log', 'a')
  os.dup2(stderr_fd.fileno(), stderr_fileno)
  for ind, elem in enumerate(sample_elems):

    mol = Chem.MolFromSmiles(elem)    
    # TODO (ytz) this is a bandage solution to reorder the atoms so
    # that they're always in the same canonical order. Presumably this
    # should be correctly implemented in the future for graph mols.
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol], smiles=elem))

  stderr_fd.close()
  os.dup2(stderr_save, stderr_fileno)
  
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  
  #return np.squeeze(np.array(features), axis=1), valid_inds
  return np.array(features), valid_inds


def featurize_smiles_np(arr, featurizer, log_every_N=1000, verbose=True):
  """Featurize individual compounds in a numpy array.

  Given a featurizer that operates on individual chemical compounds
  or macromolecules, compute & add features for that compound to the
  features array
  """
  features = []
  for ind, elem in enumerate(arr.tolist()):
    mol = Chem.MolFromSmiles(elem)
    if mol:
      new_order = rdmolfiles.CanonicalRankAtoms(mol)
      mol = rdmolops.RenumberAtoms(mol, new_order)
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))

  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  features = np.squeeze(np.array(features))
  return features.reshape(-1,)


def get_user_specified_features(df, featurizer, verbose=True):
  """Extract and merge user specified features. 

  Merge features included in dataset provided by user
  into final features dataframe

  Three types of featurization here:

    1) Molecule featurization
      -) Smiles string featurization
      -) Rdkit MOL featurization
    2) Complex featurization
      -) PDB files for interacting molecules.
    3) User specified featurizations.

  """
  time1 = time.time()
  df[featurizer.feature_fields] = df[featurizer.feature_fields].apply(
      pd.to_numeric)
  X_shard = df.as_matrix(columns=featurizer.feature_fields)
  time2 = time.time()
  log("TIMING: user specified processing took %0.3f s" % (time2 - time1),
      verbose)
  return X_shard


def featurize_mol_df(df, featurizer, field, verbose=True, log_every_N=1000):
  """Featurize individual compounds in dataframe.

  Featurizes .sdf files, so the 3-D structure should be preserved
  so we use the rdkit "mol" object created from .sdf instead of smiles
  string. Some featurizers such as CoulombMatrix also require a 3-D
  structure.  Featurizing from .sdf is currently the only way to
  perform CM feautization.
  """
  sample_elems = df[field].tolist()

  features = []
  for ind, mol in enumerate(sample_elems):
    if ind % log_every_N == 0:
      log("Featurizing sample %d" % ind, verbose)
    features.append(featurizer.featurize([mol]))
  valid_inds = np.array(
      [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
  features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
  return np.squeeze(np.array(features)), valid_inds


class DataLoader(object):
  """
  Handles loading/featurizing of chemical samples (datapoints).

  Currently knows how to load csv-files/pandas-dataframes/SDF-files. Writes a
  dataframe object to disk as output.
  """

  def __init__(self,
               tasks,
               smiles_field=None,
               id_field=None,
               mol_field=None,
               featurizer=None,
               protein_field=None,
               source_field=None,               
               verbose=True,
               prot_seq_dict=None,
               log_every_n=1000,
               input_protein=True):
    """Extracts data from input as Pandas data frame"""
    if not isinstance(tasks, list):
      raise ValueError("tasks must be a list.")
    self.verbose = verbose
    self.tasks = tasks
    self.smiles_field = smiles_field
    if id_field is None:
      self.id_field = smiles_field
    else:
      self.id_field = id_field
    self.mol_field = mol_field
    self.protein_field = protein_field
    self.source_field = source_field
    self.prot_seq_dict = prot_seq_dict
    self.user_specified_features = None
    if isinstance(featurizer, UserDefinedFeaturizer):
      self.user_specified_features = featurizer.feature_fields
    self.featurizer = featurizer
    self.log_every_n = log_every_n
    self.input_protein = input_protein

  def featurize(self, input_files, data_dir=None, shard_size=8192):
    """Featurize provided files and write to specified location.
    
    For large datasets, automatically shards into smaller chunks
    for convenience.

    Parameters
    ----------
    input_files: list
      List of input filenames.
    data_dir: str
      (Optional) Directory to store featurized dataset.
    shard_size: int
      (Optional) Number of examples stored in each shard.
    """
    log("Loading raw samples now.", self.verbose)
    log("shard_size: %d" % shard_size, self.verbose)

    if not isinstance(input_files, list):
      input_files = [input_files]

    def shard_generator():
      for shard_num, shard in enumerate(
          self.get_shards(input_files, shard_size)):
        time1 = time.time()
        X, valid_inds = self.featurize_shard(shard)
        ids = shard[self.id_field].values
        ids = ids[valid_inds]
        if len(self.tasks) > 0:
          # Featurize task results iff they exist.
          y, w = convert_df_to_numpy(shard, self.tasks, self.id_field)
          # Filter out examples where featurization failed.
          y, w = (y[valid_inds], w[valid_inds])
          assert len(X) == len(ids) == len(y) == len(w)
        else:
          # For prospective data where results are unknown, it makes
          # no sense to have y values or weights.
          y, w = (None, None)
          assert len(X) == len(ids)

        time2 = time.time()
        log("TIMING: featurizing shard %d took %0.3f s" %
            (shard_num, time2 - time1), self.verbose)
        yield X, y, w, ids

    return DiskDataset.create_dataset(
        shard_generator(), data_dir, self.tasks, verbose=self.verbose)

  def get_shards(self, input_files, shard_size):
    """Stub for children classes."""
    raise NotImplementedError

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    raise NotImplementedError


class CSVLoader(DataLoader):
  """
  Handles loading of CSV files.
  """

  def get_shards(self, input_files, shard_size, verbose=True):
    """Defines a generator which returns data for each shard"""
    return load_csv_files(input_files, shard_size, verbose=verbose)

  def featurize_shard(self, shard):
    """Featurizes a shard of an input dataframe."""
    mol_features, valid_inds = featurize_smiles_df(shard, self.featurizer, field=self.smiles_field)
    if len(mol_features.shape) > 2:
      mol_features = np.squeeze(mol_features)
    if self.input_protein:
      proteins = featurize_protein(shard, field=self.protein_field, source_field=self.source_field,
        prot_seq_dict=self.prot_seq_dict)
      # Note: for ECFP with 1024 entries, mol_features is a (8192, 1024) sized array.    
      return np.concatenate((mol_features, proteins), axis=1), valid_inds
    else:
      return mol_features, valid_inds


