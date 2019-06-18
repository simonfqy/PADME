"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar, Aneesh Pappu, modified by Qingyuan Feng "
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import tempfile
import numpy as np
import pandas as pd
import itertools
import os
import random
import math
import pdb
import warnings
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from scipy.special import comb
from scipy.cluster.hierarchy import single, fcluster
import deepchem as dc
from collections import OrderedDict
from dcCustom.data import DiskDataset
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.data import NumpyDataset
from deepchem.utils.save import load_data
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols


def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold


def randomize_arrays(array_list):
  # assumes that every array is of the same dimension
  num_rows = array_list[0].shape[0]
  perm = np.random.permutation(num_rows)
  permuted_arrays = []
  for array in array_list:
    permuted_arrays.append(array[perm])
  return permuted_arrays


class Splitter(object):
  """
    Abstract base class for chemically aware splits..
    """

  def __init__(self, verbose=False, split_cold=False, cold_drug=False, cold_target=False,
    prot_seq_dict=None, split_warm=False, cold_drug_cluster=False, threshold=0, 
    oversampled=False, input_protein=True, do_cv=False, total_folds=None, current_fold_ind=0,
    remove_val_set_entries=False, save_dir_val_set=None):
    """Creates splitter object."""
    self.verbose = verbose
    self.split_cold = split_cold
    self.cold_drug = cold_drug
    self.cold_target = cold_target
    self.split_warm = split_warm
    self.cold_drug_cluster = cold_drug_cluster
    self.prot_seq_dict = prot_seq_dict
    self.threshold = threshold
    self.oversampled = oversampled
    self.input_protein = input_protein
    self.do_cv = do_cv
    self.total_folds = total_folds
    self.current_fold_ind = current_fold_ind    
    self.remove_val_set_entries=remove_val_set_entries
    self.save_dir_val_set = save_dir_val_set


  def k_fold_split(self, dataset, k, directories=None, **kwargs):
    """
    Parameters
    ----------
    dataset: Dataset
    Dataset to do a k-fold split

    k: int
    number of folds

    directories: list of str
    list of length 2*k filepaths to save the result disk-datasets

    kwargs

    Returns
    -------
    list of length k tuples of (train, cv)

    """
    """
    :param dataset:
    :param k:
    :param directories:
    :param kwargs:
    :return: list of length k tuples of (train, cv)
    """
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(2 * k)]
    else:
      assert len(directories) == 2 * k
    cv_datasets = []
    train_ds_base = None
    train_datasets = []
    # rem_dataset is remaining portion of dataset
    if isinstance(dataset, DiskDataset):
      rem_dataset = dataset
    else:
      rem_dataset = DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                           dataset.ids)
    self.do_cv = True 
    self.total_folds = k
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      self.current_fold_ind = fold
      frac_fold = 1. / (k - fold)
      train_dir, cv_dir = directories[2 * fold], directories[2 * fold + 1]      
      fold_inds, rem_inds, _ = self.split(
          rem_dataset,
          frac_train=frac_fold,
          frac_valid=1 - frac_fold,
          frac_test=0)
      self.split_warm = False # self.split_warm is only useful in the first time.
      self.threshold = 0 # Filtering is done after the first split.
      cv_dataset = rem_dataset.select(fold_inds, select_dir=cv_dir)
      rem_dataset = rem_dataset.select(rem_inds)

      train_ds_to_merge = filter(lambda x: x is not None,
                                 [train_ds_base, rem_dataset])
      train_ds_to_merge = filter(lambda x: len(x) > 0, train_ds_to_merge)
      train_dataset = DiskDataset.merge(train_ds_to_merge, merge_dir=train_dir)
      train_datasets.append(train_dataset)

      update_train_base_merge = filter(lambda x: x is not None,
                                       [train_ds_base, cv_dataset])
      train_ds_base = DiskDataset.merge(update_train_base_merge)
      if self.oversampled:
        cv_dataset = cv_dataset.get_unique_pairs()
      if self.remove_val_set_entries:    
        print('length before removal: ', len(cv_dataset.X))    
        cv_dataset.remove_validation_set_entries(save_dir_val_set=self.save_dir_val_set) 
        print('length after removal: ', len(cv_dataset.X))           
             
      cv_datasets.append(cv_dataset)
    return list(zip(train_datasets, cv_datasets))

  # HACK: I only did oversampled dataset handling for k-fold split.
  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True):
    """
        Splits self into train/validation/test sets.

        Returns Dataset objects.
        """
    log("Computing train/valid/test indices", self.verbose)
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n)
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    train_dataset = dataset.select(train_inds, train_dir)
    # if frac_valid != 0:
    #   valid_dataset = dataset.select(valid_inds, valid_dir)
    # else:
    #   valid_dataset = None
    valid_dataset = dataset.select(valid_inds, valid_dir)
    test_dataset = dataset.select(test_inds, test_dir)

    return train_dataset, valid_dataset, test_dataset

  def train_test_split(self,
                       dataset,
                       train_dir=None,
                       test_dir=None,
                       seed=None,
                       frac_train=.8,
                       verbose=True):
    """
        Splits self into train/test sets.
        Returns Dataset objects.
        """
    valid_dir = tempfile.mkdtemp()
    train_dataset, _, test_dataset = self.train_valid_test_split(
        dataset,
        train_dir,
        valid_dir,
        test_dir,
        frac_train=frac_train,
        frac_test=1 - frac_train,
        frac_valid=0.,
        verbose=verbose)
    return train_dataset, test_dataset

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=None,
            verbose=False):
    """
        Stub to be filled in by child classes.
        """
    raise NotImplementedError


class RandomGroupSplitter(Splitter):

  def __init__(self, groups, *args, **kwargs):
    """
    A splitter class that splits on groupings. An example use case is when there
    are multiple conformations of the same molecule that share the same topology.
    This splitter subsequently guarantees that resulting splits preserve groupings.

    Note that it doesn't do any dynamic programming or something fancy to try to
    maximize the choice such that frac_train, frac_valid, or frac_test is maximized.
    It simply permutes the groups themselves. As such, use with caution if the number
    of elements per group varies significantly.

    Parameters
    ----------
    groups: array like list of hashables
      An auxiliary array indicating the group of each item.

    Eg:
    g: 3 2 2 0 1 1 2 4 3
    X: 0 1 2 3 4 5 6 7 8

    Eg:
    g: a b b e q x a a r
    X: 0 1 2 3 4 5 6 7 8

    """
    self.groups = groups
    super(RandomGroupSplitter, self).__init__(*args, **kwargs)

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):

    assert len(self.groups) == dataset.X.shape[0]
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

    if not seed is None:
      np.random.seed(seed)

    # dict is needed in case groups aren't strictly flattened or
    # hashed by something non-integer like
    group_dict = {}
    for idx, g in enumerate(self.groups):
      if g not in group_dict:
        group_dict[g] = []
      group_dict[g].append(idx)

    group_idxs = []
    for g in group_dict.values():
      group_idxs.append(g)

    group_idxs = np.array(group_idxs)

    num_groups = len(group_idxs)
    train_cutoff = int(frac_train * num_groups)
    valid_cutoff = int((frac_train + frac_valid) * num_groups)
    shuffled_group_idxs = np.random.permutation(range(num_groups))

    train_groups = shuffled_group_idxs[:train_cutoff]
    valid_groups = shuffled_group_idxs[train_cutoff:valid_cutoff]
    test_groups = shuffled_group_idxs[valid_cutoff:]

    train_idxs = list(itertools.chain(*group_idxs[train_groups]))
    valid_idxs = list(itertools.chain(*group_idxs[valid_groups]))
    test_idxs = list(itertools.chain(*group_idxs[test_groups]))

    return train_idxs, valid_idxs, test_idxs


class RandomStratifiedSplitter(Splitter):
  """
    RandomStratified Splitter class.

    For sparse multitask datasets, a standard split offers no guarantees that the
    splits will have any activate compounds. This class guarantees that each task
    will have a proportional split of the activates in a split. TO do this, a
    ragged split is performed with different numbers of compounds taken from each
    task. Thus, the length of the split arrays may exceed the split of the
    original array. That said, no datapoint is copied to more than one split, so
    correctness is still ensured.

    Note that this splitter is only valid for boolean label data.

    TODO(rbharath): This splitter should be refactored to match style of other
    splitter classes.
    """

  def __generate_required_hits(self, w, frac_split):
    # returns list of per column sum of non zero elements
    required_hits = (w != 0).sum(axis=0)
    for col_hits in required_hits:
      col_hits = int(frac_split * col_hits)
    return required_hits

  def get_task_split_indices(self, y, w, frac_split):
    """Returns num datapoints needed per task to split properly."""
    w_present = (w != 0)
    y_present = y * w_present

    # Compute number of actives needed per task.
    task_actives = np.sum(y_present, axis=0)
    task_split_actives = (frac_split * task_actives).astype(int)

    # loop through each column and obtain index required to splice out for
    # required fraction of hits
    split_indices = []
    n_tasks = np.shape(y)[1]
    for task in range(n_tasks):
      actives_count = task_split_actives[task]
      cum_task_actives = np.cumsum(y_present[:, task])
      # Find the first index where the cumulative number of actives equals
      # the actives_count
      split_index = np.amin(np.where(cum_task_actives >= actives_count)[0])
      # Note that np.where tells us last index required to exceed
      # actives_count, so we actually want the following location
      split_indices.append(split_index + 1)
    return split_indices

    # TODO(rbharath): Refactor this split method to match API of other splits (or

  # potentially refactor those to match this.

  def split(self, dataset, frac_split, split_dirs=None):
    """
        Method that does bulk of splitting dataset.
        """
    if split_dirs is not None:
      assert len(split_dirs) == 2
    else:
      split_dirs = [tempfile.mkdtemp(), tempfile.mkdtemp()]

    # Handle edge case where frac_split is 1
    if frac_split == 1:
      dataset_1 = NumpyDataset(dataset.X, dataset.y, dataset.w, dataset.ids)
      dataset_2 = None
      return dataset_1, dataset_2
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w,
                                     dataset.ids))
    split_indices = self.get_task_split_indices(y, w, frac_split)

    # Create weight matrices fpor two haves.
    w_1, w_2 = np.zeros_like(w), np.zeros_like(w)
    for task, split_index in enumerate(split_indices):
      # copy over up to required index for weight first_split
      w_1[:split_index, task] = w[:split_index, task]
      w_2[split_index:, task] = w[split_index:, task]

    # check out if any rows in either w_1 or w_2 are just zeros
    rows_1 = w_1.any(axis=1)
    X_1, y_1, w_1, ids_1 = X[rows_1], y[rows_1], w_1[rows_1], ids[rows_1]
    dataset_1 = NumpyDataset(X_1, y_1, w_1, ids_1)

    rows_2 = w_2.any(axis=1)
    X_2, y_2, w_2, ids_2 = X[rows_2], y[rows_2], w_2[rows_2], ids[rows_2]
    dataset_2 = NumpyDataset(X_2, y_2, w_2, ids_2)

    return dataset_1, dataset_2

  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000):
    """Custom split due to raggedness in original split.
        """
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()
    # Obtain original x, y, and w arrays and shuffle
    X, y, w, ids = randomize_arrays((dataset.X, dataset.y, dataset.w,
                                     dataset.ids))
    rem_dir = tempfile.mkdtemp()
    train_dataset, rem_dataset = self.split(dataset, frac_train,
                                            [train_dir, rem_dir])

    # calculate percent split for valid (out of test and valid)
    if frac_valid + frac_test > 0:
      valid_percentage = frac_valid / (frac_valid + frac_test)
    else:
      return train_dataset, None, None
    # split test data into valid and test, treating sub test set also as sparse
    valid_dataset, test_dataset = self.split(dataset, valid_percentage,
                                             [valid_dir, test_dir])

    return train_dataset, valid_dataset, test_dataset

  def k_fold_split(self, dataset, k, directories=None, **kwargs):
    """Needs custom implementation due to ragged splits for stratification."""
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k
    fold_datasets = []
    # rem_dataset is remaining portion of dataset
    rem_dataset = dataset
    for fold in range(k):
      # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
      # to k-1.
      frac_fold = 1. / (k - fold)
      fold_dir = directories[fold]
      rem_dir = tempfile.mkdtemp()
      fold_dataset, rem_dataset = self.split(rem_dataset, frac_fold,
                                             [fold_dir, rem_dir])
      fold_datasets.append(fold_dataset)
    return fold_datasets


class SingletaskStratifiedSplitter(Splitter):
  """
    Class for doing data splits by stratification on a single task.

    Example:

    >>> n_samples = 100
    >>> n_features = 10
    >>> n_tasks = 10
    >>> X = np.random.rand(n_samples, n_features)
    >>> y = np.random.rand(n_samples, n_tasks)
    >>> w = np.ones_like(y)
    >>> dataset = DiskDataset.from_numpy(np.ones((100,n_tasks)), np.ones((100,n_tasks)), verbose=False)
    >>> splitter = SingletaskStratifiedSplitter(task_number=5, verbose=False)
    >>> train_dataset, test_dataset = splitter.train_test_split(dataset)

    """

  def __init__(self, task_number=0, verbose=False):
    """
        Creates splitter object.

        Parameters
        ----------
        task_number: int (Optional, Default 0)
          Task number for stratification.
        verbose: bool (Optional, Default False)
          Controls logging frequency.
        """
    self.task_number = task_number
    self.verbose = verbose

  def k_fold_split(self,
                   dataset,
                   k,
                   directories=None,
                   seed=None,
                   log_every_n=None,
                   **kwargs):
    """
        Splits compounds into k-folds using stratified sampling.
        Overriding base class k_fold_split.

        Parameters
        ----------
        dataset: dc.data.Dataset object
          Dataset.
        k: int
          Number of folds.
        seed: int (Optional, Default None)
          Random seed.
        log_every_n: int (Optional, Default None)
          Log every n examples (not currently used).

        Returns
        -------
        fold_datasets: List
          List containing dc.data.Dataset objects
        """
    log("Computing K-fold split", self.verbose)
    if directories is None:
      directories = [tempfile.mkdtemp() for _ in range(k)]
    else:
      assert len(directories) == k

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)
    sortidx_list = np.array_split(sortidx, k)

    fold_datasets = []
    for fold in range(k):
      fold_dir = directories[fold]
      fold_ind = sortidx_list[fold]
      fold_dataset = dataset.select(fold_ind, fold_dir)
      fold_datasets.append(fold_dataset)
    return fold_datasets

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits compounds into train/validation/test using stratified sampling.

        Parameters
        ----------
        dataset: dc.data.Dataset object
          Dataset.
        seed: int (Optional, Default None)
          Random seed.
        frac_train: float (Optional, Default .8)
          Fraction of dataset put into training data.
        frac_valid: float (Optional, Default .1)
          Fraction of dataset put into validation data.
        frac_test: float (Optional, Default .1)
          Fraction of dataset put into test data.
        log_every_n: int (Optional, Default None)
          Log every n examples (not currently used).

        Returns
        -------
        retval: Tuple
          Tuple containing train indices, valid indices, and test indices
        """
    # JSG Assert that split fractions can be written as proper fractions over 10.
    # This can be generalized in the future with some common demoninator determination.
    # This will work for 80/20 train/test or 80/10/10 train/valid/test (most use cases).
    np.testing.assert_equal(frac_train + frac_valid + frac_test, 1.)
    np.testing.assert_equal(10 * frac_train + 10 * frac_valid + 10 * frac_test,
                            10.)

    if not seed is None:
      np.random.seed(seed)

    y_s = dataset.y[:, self.task_number]
    sortidx = np.argsort(y_s)

    split_cd = 10
    train_cutoff = int(np.round(frac_train * split_cd))
    valid_cutoff = int(np.round(frac_valid * split_cd)) + train_cutoff
    test_cutoff = int(np.round(frac_test * split_cd)) + valid_cutoff

    train_idx = np.array([])
    valid_idx = np.array([])
    test_idx = np.array([])

    while sortidx.shape[0] >= split_cd:
      sortidx_split, sortidx = np.split(sortidx, [split_cd])
      shuffled = np.random.permutation(range(split_cd))
      train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
      valid_idx = np.hstack(
          [valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
      test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

    # Append remaining examples to train
    if sortidx.shape[0] > 0: np.hstack([train_idx, sortidx])

    return (train_idx, valid_idx, test_idx)


class MolecularWeightSplitter(Splitter):
  """
    Class for doing data splits by molecular weight.
    """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds into train/validation/test using the MW calculated
        by SMILES string.
        """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
      np.random.seed(seed)

    mws = []
    for smiles in dataset.ids:
      mol = Chem.MolFromSmiles(smiles)
      mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
      mws.append(mw)

    # Sort by increasing MW
    mws = np.array(mws)
    sortidx = np.argsort(mws)

    train_cutoff = frac_train * len(sortidx)
    valid_cutoff = (frac_train + frac_valid) * len(sortidx)

    return (sortidx[:train_cutoff], sortidx[train_cutoff:valid_cutoff],
            sortidx[valid_cutoff:])


class MaxMinSplitter(Splitter):
  """
  Class for doing splits based on the MaxMin diversity algorithm. Intuitively,
  the test set is comprised of the most diverse compounds of the entire dataset.
  Furthermore, the validation set is comprised of diverse compounds under
  the test set.
  """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits internal compounds randomly into train/validation/test.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if seed is None:
      seed = random.randint(0, 2**30)
    np.random.seed(seed)

    num_datapoints = len(dataset)

    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)

    num_train = train_cutoff
    num_valid = valid_cutoff - train_cutoff
    num_test = num_datapoints - valid_cutoff

    all_mols = []
    for ind, smiles in enumerate(dataset.ids):
      all_mols.append(Chem.MolFromSmiles(smiles))

    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in all_mols]

    def distance(i, j):
      return 1 - DataStructs.DiceSimilarity(fps[i], fps[j])

    picker = MaxMinPicker()
    testIndices = picker.LazyPick(
        distFunc=distance,
        poolSize=num_datapoints,
        pickSize=num_test,
        seed=seed)

    validTestIndices = picker.LazyPick(
        distFunc=distance,
        poolSize=num_datapoints,
        pickSize=num_valid + num_test,
        firstPicks=testIndices,
        seed=seed)

    allSet = set(range(num_datapoints))
    testSet = set(testIndices)
    validSet = set(validTestIndices) - testSet

    trainSet = allSet - testSet - validSet

    assert len(testSet & validSet) == 0
    assert len(testSet & trainSet) == 0
    assert len(validSet & trainSet) == 0
    assert (validSet | trainSet | testSet) == allSet

    return sorted(list(trainSet)), sorted(list(validSet)), sorted(list(testSet))


class RandomSplitter(Splitter):
  """
    Class for doing random data splits.
  """
  def read_data(self, dataset, mol_entries, prot_entries=None, 
    entry_id_to_pair=None, pair_to_entry_id=None):

    entry_id = 0   
    entry_id_to_mol = dict()
    mol_to_entry_id = dict() 
    entry_id_to_assay = dict()
    assay_to_entry_id = dict()
    n_tasks = np.shape(dataset.y)[1]

    for (X, y, w, _) in dataset.itersamples():    
      
      mol = X[0]
      if mol not in mol_entries:
        mol_entries[mol] = set()
      mol_entries[mol].add(entry_id)
      
      if self.input_protein:
        prot = X[1]
        if prot not in prot_entries:
          prot_entries[prot] = set()  
        prot_entries[prot].add(entry_id)

        pair = (mol, prot)
        entry_id_to_pair[entry_id] = pair
        if pair not in pair_to_entry_id:
          pair_to_entry_id[pair] = set()
        pair_to_entry_id[pair].add(entry_id)
      else:
        # entry_id_to_mol[entry_id] = mol
        # if mol not in mol_to_entry_id:
        #   mol_to_entry_id[mol] = set()
        # mol_to_entry_id[mol].add(entry_id)
        
        available_assays_for_this_mol = [i for i in range(n_tasks) if w[i] == 1.0]
        for assay_id in available_assays_for_this_mol:
          if assay_id not in assay_to_entry_id:
            assay_to_entry_id[assay_id] = set()
          assay_to_entry_id[assay_id].add(entry_id)
        entry_id_to_assay[entry_id] = set(available_assays_for_this_mol)

      entry_id += 1      

    print("last entry_id: ", entry_id)
    print("len(mol_entries): ", len(mol_entries))
    if self.input_protein:
      print("len(prot_entries): ", len(prot_entries))

    return entry_id_to_mol, mol_to_entry_id, entry_id_to_assay, assay_to_entry_id


  def remove_entities(self, entity_to_entry_id, removed_entries):
    counter = 0        
    for entity in list(entity_to_entry_id):      
      entity_to_entry_id[entity].difference_update(removed_entries) 
      if len(entity_to_entry_id[entity]) <= self.threshold:
        counter += 1
        removed_entries.update(entity_to_entry_id[entity])            
        del entity_to_entry_id[entity]            

    return counter


  def filter(self, mol_entries, all_entry_id, removed_entries, frac_train, frac_test, prot_entries=None):
    
    carry_on = True
    num_already_removed = 0

    while carry_on:      
      counter = self.remove_entities(mol_entries, removed_entries)      
      num_newly_removed = len(removed_entries) - num_already_removed
      num_already_removed = len(removed_entries)
      print("removing %d molecules with no more than %d entries, resulting in the \
        removal of %d entries." % (counter, self.threshold, num_newly_removed))
      
      if self.input_protein:        
        counter = self.remove_entities(prot_entries, removed_entries) 
        num_newly_removed = len(removed_entries) - num_already_removed
        num_already_removed = len(removed_entries)
        print("removing %d proteins with no more than %d entries, resulting in the \
          removal of %d entries." % (counter, self.threshold, num_newly_removed))

      carry_on = False
      for _, this_mol_entries in mol_entries.items():
        remain_this_mol_entries = this_mol_entries - removed_entries
        if len(remain_this_mol_entries) <= self.threshold:
          carry_on = True
          break

    all_entry_id.difference_update(removed_entries) 
    num_datapoints = len(all_entry_id)
    num_training = int(num_datapoints * frac_train)     
    num_test = int(num_datapoints * frac_test)
    num_validation = num_datapoints - num_training - num_test
    print("After the filtering")
    print("Number of entries removed: ", len(removed_entries))
    print("Number of entries remaining: ", len(all_entry_id))
    print("len(mol_entries): ", len(mol_entries))
    if self.input_protein:
      print("len(prot_entries): ", len(prot_entries))

    return num_training, num_validation


  def randomly_cold_split(self, mol_entries, prot_entries, entries_for_training, num_training):
    # We need to split the dataset into cold-drugs and cold-targets. 
    while True:
      num_drug_remain = len(mol_entries)
      num_prot_remain = len(prot_entries)
      rand_num = random.uniform(0, num_drug_remain + num_prot_remain)
      # Let the random number decide which entity to choose, drug or proteins.
      entity_collection = mol_entries if rand_num <= num_drug_remain else prot_entries        
      entity_chosen = random.choice(list(entity_collection.keys()))
      print("num_training: ", num_training)
      print("len(entries_for_training): ", len(entries_for_training))
      print("length of new elements: ", len(entity_collection[entity_chosen] - 
        entries_for_training))     

      if len(entries_for_training.union(entity_collection[entity_chosen])) > num_training:
        new_elements = entity_collection[entity_chosen].difference(entries_for_training)
        num_to_choose = num_training - len(entries_for_training)
        new_elements = random.sample(new_elements, num_to_choose)
        entries_for_training.update(new_elements)        
      else:
        # Only take union if the total size is within the limit.
        entries_for_training.update(entity_collection[entity_chosen])
        del entity_collection[entity_chosen]
        
      if len(entries_for_training) >= num_training:
        break    
    return 


  def assign_training_or_preserved_entries(self, prots_without_entries_of_type_we_wanna_assign, 
    prots_without_entries_of_the_opposite_type, prots_whose_current_mol_entry_assigned, 
    warm_prots, entries_of_this_mol, prot_entries, entries_with_type_we_wanna_assign):
    unsolved = True
    # Select the training entries for those proteins that have no training entries.
    if len(prots_without_entries_of_type_we_wanna_assign) > 0:            
      selected_prot = random.sample(prots_without_entries_of_type_we_wanna_assign, 1)[0]            
      unsolved = False            
    if unsolved and len(warm_prots) > 0:            
      selected_prot = random.sample(warm_prots, 1)[0]
      unsolved = False
    if unsolved and len(prots_without_entries_of_the_opposite_type) > 0:
      candidate_prots = set()
      for protein in prots_without_entries_of_the_opposite_type:
        remaining_unassigned_entries_for_this_prot = prot_entries[protein] - entries_with_type_we_wanna_assign
        # If there are more than 1 entries unassigned for the current protein, we can assign it the desired
        # type, since there are still chances to make it "warm".
        if len(remaining_unassigned_entries_for_this_prot) > 1:
          candidate_prots.add(protein)
      if len(candidate_prots) > 0:
        selected_prot = random.sample(candidate_prots, 1)[0]
        unsolved = False
    assert not unsolved and selected_prot is not None
    entry_to_write = entries_of_this_mol & prot_entries[selected_prot]
    assert len(entry_to_write) == 1
    prots_whose_current_mol_entry_assigned.add(selected_prot)
    prots_without_entries_of_type_we_wanna_assign.difference_update(prots_whose_current_mol_entry_assigned)
    warm_prots.difference_update(prots_whose_current_mol_entry_assigned)
    prots_without_entries_of_the_opposite_type.difference_update(prots_whose_current_mol_entry_assigned) 
    entries_with_type_we_wanna_assign.update(entry_to_write)


  def eliminate_cold_protein(self, prots_without_preserved_entries, prots_without_training_entries,
    prots_whose_current_mol_entry_assigned, num_validation, num_training, entries_preserved, 
    entries_for_training, this_mol_entries, prot_entries):  
    # Eliminate the cold proteins among the proteins that have unassigned intersection  
    # entries with the current molecule.         
    double_cold_prots = prots_without_preserved_entries & prots_without_training_entries            
    for protein in double_cold_prots:
      num_to_preserve = num_validation - len(entries_preserved)
      num_to_train = num_training - len(entries_for_training)
      entry_to_write = this_mol_entries & prot_entries[protein]
      assert len(entry_to_write) == 1
      rand_num = random.uniform(0, num_to_preserve + num_to_train)            
      if rand_num <= num_to_train:              
        entries_for_training.update(entry_to_write)
      else:
        entries_preserved.update(entry_to_write)
      prots_whose_current_mol_entry_assigned.add(protein)
    
    prots_without_preserved_entries.difference_update(prots_whose_current_mol_entry_assigned) 
    prots_without_training_entries.difference_update(prots_whose_current_mol_entry_assigned) 
    
    # Now deal with single cold proteins.          
    if len(prots_without_preserved_entries) > 0:            
      for protein in prots_without_preserved_entries:
        entry_to_write = this_mol_entries & prot_entries[protein]
        entries_preserved.update(entry_to_write)
        prots_whose_current_mol_entry_assigned.add(protein)                        
    if len(prots_without_training_entries) > 0:
      for protein in prots_without_training_entries:
        entry_to_write = this_mol_entries & prot_entries[protein]
        entries_for_training.update(entry_to_write)
        prots_whose_current_mol_entry_assigned.add(protein)


  def warm_split_with_input_protein(self, mol_entries, prot_entries, removed_entries, 
    entries_for_training, entry_id_to_pair, num_validation, num_training): 
    assert self.input_protein
    # The word "preserve" here actually means "preserved for other CV splits", in other words,
    # it is referring to the validation fold for the current split.
    entries_preserved = set()
    unassigned_entries = set()      
    for _, this_mol_entries in mol_entries.items():
      remain_this_mol_entries = this_mol_entries - removed_entries
      assert len(remain_this_mol_entries) > self.threshold
      assert remain_this_mol_entries == this_mol_entries        
      mol_warm = False
      # The next 3 protein sets are only intended for those proteins whose intersections with the
      # current molecule are unassigned.
      warm_prots = set()
      prots_without_training_entries = set()
      prots_without_preserved_entries = set() 
      # The set of proteins whose intersections with the current molecule are assigned.       
      prots_whose_current_mol_entry_assigned = set()
      # For every molecule, we need to make sure not only itself has both test and training fold
      # splits, but the proteins that co-ocurr with them have both as well.                 
      this_mol_has_no_training_entries = (len(entries_for_training & this_mol_entries) <= 0)      
      this_mol_has_no_preserved_entries = (len(entries_preserved & this_mol_entries) <= 0)
        
      for entry in this_mol_entries:
        protein = entry_id_to_pair[entry][1]
        assert entry in prot_entries[protein]
        if entry in entries_preserved or entry in entries_for_training:
          prots_whose_current_mol_entry_assigned.add(protein)
        if len(prot_entries[protein] & entries_for_training) <= 0 and protein not in \
          prots_whose_current_mol_entry_assigned:            
          prots_without_training_entries.add(protein)
        if len(prot_entries[protein] & entries_preserved) <= 0 and protein not in \
          prots_whose_current_mol_entry_assigned:
          prots_without_preserved_entries.add(protein)
        if len(entries_preserved & prot_entries[protein]) > 0 and len(entries_for_training &
          prot_entries[protein]) > 0 and protein not in prots_whose_current_mol_entry_assigned:
          warm_prots.add(protein)
         
      if this_mol_has_no_training_entries:
        self.assign_training_or_preserved_entries(prots_without_training_entries, 
          prots_without_preserved_entries, prots_whose_current_mol_entry_assigned, 
          warm_prots, this_mol_entries, prot_entries, entries_for_training)                
      
      if this_mol_has_no_preserved_entries:
        self.assign_training_or_preserved_entries(prots_without_preserved_entries, 
          prots_without_training_entries, prots_whose_current_mol_entry_assigned, 
          warm_prots, this_mol_entries, prot_entries, entries_preserved) 
                
      if len(entries_preserved & this_mol_entries) > 0 and len(entries_for_training &
        this_mol_entries) > 0:
        mol_warm = True
      
      assert mol_warm
      # Now handle the warm molecules. Objective: make all the entries' proteins warm if possible.        
      if len(warm_prots.union(prots_whose_current_mol_entry_assigned)) == len(this_mol_entries):
        # All entries of this molecule are either assigned, or unassigned but the corresponding protein is warm.
        unassigned_entries.update(this_mol_entries - entries_preserved - entries_for_training) 
        continue
      # Fill them randomly.
      assert len(entries_preserved & entries_for_training) <= 0  
      
      self.eliminate_cold_protein(prots_without_preserved_entries, prots_without_training_entries,
        prots_whose_current_mol_entry_assigned, num_validation, num_training, entries_preserved, 
        entries_for_training, this_mol_entries, prot_entries)           
      
      # Make sure that all the entries of this molecule are either assigned, or unassigned but the 
      # corresponding protein is warm.
      assert len(warm_prots.union(prots_whose_current_mol_entry_assigned)) == len(this_mol_entries)
      unassigned_entries.update(this_mol_entries - entries_preserved - entries_for_training)

    # Next need to make sure the corresponding proteins are warm-splitted.
    for _, this_prot_entries in prot_entries.items():
      remain_this_prot_entries = this_prot_entries - removed_entries
      assert len(remain_this_prot_entries) > self.threshold
      assert remain_this_prot_entries == this_prot_entries
      assert len(this_prot_entries & entries_for_training) > 0
      assert len(this_prot_entries & entries_preserved) > 0

    num_to_select = num_training - len(entries_for_training)
    assert len(unassigned_entries) >= num_to_select
    entries_for_training.update(random.sample(unassigned_entries, num_to_select))


  def cold_drug_split_without_input_protein(self, all_entry_id, entries_for_training, 
    entry_id_to_assay, assay_to_entry_id, num_training, num_validation,
    num_test, frac_train):
    # This function handles the cold drug split while keeping the proteins "warm" 
    assert not self.input_protein
    assay_id_list = list(assay_to_entry_id)  
    fully_allocated = False
    # Start from the assays that have the least amount of entries corresponding to them,
    # since they easily become "cold", which means all of them are absent in some folds
    # (in this case, all assigned in entries_for_training or none are). 
    num_entry_list_corresponding_to_assay_id = [len(assay_to_entry_id[assay]) for assay in assay_id_list]
    ind_of_assay_id_list_in_sorted_order = np.argsort(np.array(num_entry_list_corresponding_to_assay_id))

    # This loop ensures that all the assays (proteins) are "warm".
    for ind_of_assay_id_list in ind_of_assay_id_list_in_sorted_order:
      assay_id_to_choose = assay_id_list[ind_of_assay_id_list]
      num_entries_of_this_assay = num_entry_list_corresponding_to_assay_id[ind_of_assay_id_list]
      num_entries_of_this_assay_for_training = int(num_entries_of_this_assay * frac_train)
      # Make sure that it is not "cold".
      #assert num_entries_of_this_assay_for_training > 0
      entries_of_this_assay_already_selected = entries_for_training & assay_to_entry_id[assay_id_to_choose]      
      num_additional_entries = num_entries_of_this_assay_for_training - len(entries_of_this_assay_already_selected)
      if num_additional_entries <= 0:
        continue
      entries_available = assay_to_entry_id[assay_id_to_choose] - entries_for_training
      entries_for_training.update(set(random.sample(entries_available, num_additional_entries)))

      if len(entries_for_training) >= num_training:
        fully_allocated = True
        break   

    potential_candidate_entries = set(all_entry_id) - entries_for_training
    while not fully_allocated:
      candidate_entry = random.sample(potential_candidate_entries, 1)[0]
      related_assays = entry_id_to_assay[candidate_entry]
      # Check whether removing the current candidate_entry will cause any problems.
      num_unassigned_entry_for_related_assays = [len(assay_to_entry_id[r_assay] - entries_for_training) for 
        r_assay in related_assays]
      potential_candidate_entries.remove(candidate_entry)
      if self.do_cv:
        if min(num_unassigned_entry_for_related_assays) < self.total_folds - self.current_fold_ind:
          continue
      elif min(num_unassigned_entry_for_related_assays) < (num_training > 0) + (num_validation > 0) + \
        (num_test > 0):
        # Not doing cross-validation, doing simple train-val-test split.         
        continue
      
      entries_for_training.add(candidate_entry)

      if len(entries_for_training) >= num_training:
        fully_allocated = True   

    # We need to make sure the proteins/assays are not "cold", i.e., they are all present in every fold.
    # for _, entry_id_set in assay_to_entry_id.items():
    #   assert len(entry_id_set & entries_for_training) > 0


  def cold_drug_or_target_split(self, mol_entries, prot_entries, num_training, entries_for_training):
    entity_entries = {}
    if self.cold_drug:
      entity_entries = mol_entries
    elif self.cold_target:
      entity_entries = prot_entries         
    
    print("len(entity_entries): ", len(entity_entries))
     
    while True:        
      entity_chosen = random.choice(list(entity_entries.keys()))
      
      if num_training - len(entries_for_training) < 20:
        print("num_training: ", num_training)
        print("len(entries_for_training): ", len(entries_for_training))
        print("length of new elements: ", len(entity_entries[entity_chosen] - 
          entries_for_training))
                    
      if len(entries_for_training.union(entity_entries[entity_chosen])) > num_training:
        new_elements = entity_entries[entity_chosen].difference(entries_for_training)
        num_to_choose = num_training - len(entries_for_training)
        new_elements = random.sample(new_elements, num_to_choose)
        entries_for_training.update(new_elements)        
      else:
        # Only take union if the total size is within the limit.
        entries_for_training.update(entity_entries[entity_chosen])
        del entity_entries[entity_chosen]
        
      if len(entries_for_training) >= num_training:
        break


  def cold_drug_cluster_split(self, mol_entries, num_training, entries_for_training):
    if not hasattr(self, 'mol_to_fp'):
      self.mol_to_fp = dict()
    if not hasattr(self, 'mols_to_dist'):
      self.mols_to_dist = dict()
    mol_list = list(mol_entries)
    for mol in mol_entries:
      if mol in self.mol_to_fp:
        continue
      mol_obj = Chem.MolFromSmiles(mol.smiles)
      fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_obj, 2, nBits=1024)
      self.mol_to_fp[mol] = fingerprint

    num_mols = len(mol_entries)
    len_vector = int(num_mols*(num_mols-1)/2)
    distance_arr = np.empty(len_vector)
    distance_arr[:] = np.nan

    for i, mol_1 in enumerate(mol_list):
      for j in range(i + 1, len(mol_list)):
        mol_2 = mol_list[j]        
        mols_set = frozenset([mol_1, mol_2])
        index = int(comb(num_mols, 2)) - int(comb(num_mols - i, 2)) + j - i - 1
        if mols_set in self.mols_to_dist:
          distance_arr[index] = self.mols_to_dist[mols_set]
          continue
        sim = DataStructs.FingerprintSimilarity(self.mol_to_fp[mol_1], self.mol_to_fp[mol_2])        
        distance_arr[index] = 1 - sim
        self.mols_to_dist[mols_set] = 1 - sim

    clustering = fcluster(single(distance_arr), t=0.3, criterion='distance')
    cluster_id_to_mol_ind = {}
    for i, cluster_id in enumerate(clustering):
      if cluster_id not in cluster_id_to_mol_ind:
        cluster_id_to_mol_ind[cluster_id] = set()
      cluster_id_to_mol_ind[cluster_id].add(i)

    print("Number of molecules: ", num_mols)
    print("Number of clusters: ", len(cluster_id_to_mol_ind))
    
    while True:        
      cluster_id_chosen = random.choice(list(cluster_id_to_mol_ind))
      entries_of_this_cluster = set()
      for mol_ind in cluster_id_to_mol_ind[cluster_id_chosen]:
        entries_of_this_cluster.update(mol_entries[mol_list[mol_ind]])
      
      if num_training - len(entries_for_training) < 20:
        print("num_training: ", num_training)
        print("len(entries_for_training): ", len(entries_for_training))
        print("length of new elements: ", len(entries_of_this_cluster - 
          entries_for_training))
                    
      if len(entries_for_training.union(entries_of_this_cluster)) > num_training:
        new_elements = entries_of_this_cluster.difference(entries_for_training)
        num_to_choose = num_training - len(entries_for_training)
        new_elements = random.sample(new_elements, num_to_choose)
        entries_for_training.update(new_elements)        
      else:
        # Only take union if the total size is within the limit.
        entries_for_training.update(entries_of_this_cluster)
        mol_ind_to_remove = set()
        for mol_ind in cluster_id_to_mol_ind[cluster_id_chosen]:
          del mol_entries[mol_list[mol_ind]]
          mol_ind_to_remove.add(mol_ind)
        cluster_id_to_mol_ind[cluster_id_chosen].difference_update(mol_ind_to_remove)
        if len(cluster_id_to_mol_ind[cluster_id_chosen]) <= 0:
          del cluster_id_to_mol_ind[cluster_id_chosen]
        
      if len(entries_for_training) >= num_training:
        break


  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds randomly into train/validation/test.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    if not seed is None:
      np.random.seed(seed)
      random.seed(seed)
    num_datapoints = len(dataset)

    assert (self.split_cold + self.cold_drug + self.cold_target + self.split_warm \
      + self.cold_drug_cluster) <= 1    
    
    if not (self.split_cold or self.cold_drug or self.cold_target or 
      self.split_warm or self.cold_drug_cluster or self.oversampled) and self.threshold <= 0:
      # This is the case where you are not performing filtering and doing some general random
      # splitting.
      train_cutoff = int(frac_train * num_datapoints)
      valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
      shuffled = np.random.permutation(range(num_datapoints))
      return (shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff],
              shuffled[valid_cutoff:])

    if not self.input_protein:
      assert not (self.split_cold or self.cold_target or self.split_warm)

    if self.oversampled:
      # HACK: I haven't implemented the oversampled splitting under these circumstances. It is
      # used as an early failure protection.
      assert not (self.split_cold or self.split_warm or self.cold_drug_cluster or self.threshold > 0)

    # Only execute when need to perform filtering or doing some specific splitting.
    # In warm splitting of cross-validation after the initial split, the program returns from
    # the last if-block.
    all_entry_id = set(range(num_datapoints))
    entries_for_training = set()
    num_training = int(num_datapoints * frac_train)
    num_test = int(num_datapoints * frac_test)
    num_validation = num_datapoints - num_training - num_test 
    special_splitting = True   
    mol_entries = OrderedDict() 
    removed_entries = set()
    if self.input_protein:
      prot_entries = OrderedDict()  
      entry_id_to_pair = {} 
      pair_to_entry_id = {}
    else:
      prot_entries, entry_id_to_pair, pair_to_entry_id = None, None, None

    entry_id_to_mol, mol_to_entry_id, entry_id_to_assay, assay_to_entry_id = self.read_data(
      dataset, mol_entries, prot_entries=prot_entries,
      entry_id_to_pair=entry_id_to_pair, pair_to_entry_id=pair_to_entry_id)

    if self.threshold > 0:
      # We need to filter the dataset.      
      num_training, num_validation = self.filter(mol_entries, 
        all_entry_id, removed_entries, frac_train, frac_test, prot_entries=prot_entries)
        
    # All the special splitting schemes here are intended for Cross-Validation. I have
    # not tested them on ordinary train-test splits.
    if self.split_cold:    
      self.randomly_cold_split(mol_entries, prot_entries, entries_for_training, num_training)   
      
    elif self.split_warm:
      self.warm_split_with_input_protein(mol_entries, prot_entries, removed_entries, 
        entries_for_training, entry_id_to_pair, num_validation, num_training)       
    
    elif (self.cold_drug or self.cold_target) and self.input_protein:      
      self.cold_drug_or_target_split(mol_entries, prot_entries, num_training, entries_for_training)

    elif self.cold_target:
      # This actually wouldn't happen, because a prior assert statement prevents it.
      raise ValueError("When self.input_protein == False, self.cold_target cannot be done.")

    elif self.cold_drug:
      # self.input_protein is False here.
      num_entries_list = [len(entries) for _, entries in assay_to_entry_id.items()]
      # if self.do_cv:
      #   # Make sure that "warm" proteins splitting can be achieved.        
      #   if self.current_fold_ind > 0 and (min(num_entries_list) < self.total_folds - 
      #     self.current_fold_ind):
      #     raise ValueError("Something went wrong in the %sth iteration of the CV split" % 
      #       str(self.current_fold_ind))
      #   assert min(num_entries_list) >= self.total_folds - self.current_fold_ind

      # elif min(num_entries_list) < (num_training > 0) + (num_validation > 0) + (num_test > 0):
      #   raise ValueError("Some assays have too few entries, unable to assign at least 1 entry \
      #     to training, validation/test sets.")

      self.cold_drug_split_without_input_protein(all_entry_id, entries_for_training, 
        entry_id_to_assay, assay_to_entry_id, num_training, num_validation, num_test, frac_train)

    # elif self.cold_drug_cluster and self.input_protein:
    elif self.cold_drug_cluster:
      self.cold_drug_cluster_split(mol_entries, num_training, entries_for_training)

    # elif self.cold_drug_cluster:
    #   raise ValueError("input_protein==False is not yet supported in cold_drug_cluster scenario.")
     
    elif self.oversampled:
      assert self.input_protein
      # TODO: need to rewrite this section to accommodate the self.input_protein=False scenario.
      print("len(pair_to_entry_id): ", len(pair_to_entry_id))
      while True:
        pair_chosen = random.choice(list(pair_to_entry_id.keys()))
        if len(entries_for_training.union(pair_to_entry_id[pair_chosen])) > num_training:
          new_elements = pair_to_entry_id[pair_chosen].difference(entries_for_training)
          num_to_choose = num_training - len(entries_for_training)
          new_elements = random.sample(new_elements, num_to_choose)
          entries_for_training.update(new_elements)  
        else:
          new_elements = pair_to_entry_id[pair_chosen].difference(entries_for_training)
          entries_for_training.update(new_elements)
          del pair_to_entry_id[pair_chosen]
        if len(entries_for_training) >= num_training:
          break

    else:
      special_splitting = False
      entries_for_training = set(random.sample(all_entry_id, num_training))

    if num_test > 0 and special_splitting:
      warnings.warn("The special splitting schemes are intended for cross validation. They might not \
        work as intended when the dataset is split between training, validation and test datasets.")

    remaining_entries = all_entry_id.difference(entries_for_training)
    entries_for_validation = set(random.sample(remaining_entries, num_validation))
    entries_for_testing = remaining_entries.difference(entries_for_validation)
    
    assert len(entries_for_training & entries_for_testing) == 0
    assert len(entries_for_training & entries_for_validation) == 0
    assert len(entries_for_validation & entries_for_testing) == 0
    assert (entries_for_training | entries_for_validation | entries_for_testing) == all_entry_id
    
    return list(entries_for_training), list(entries_for_validation), list(entries_for_testing)
    

class IndexSplitter(Splitter):
  """
    Class for simple order based splits.
    """

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds into train/validation/test in provided order.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = range(num_datapoints)
    return (indices[:train_cutoff], indices[train_cutoff:valid_cutoff],
            indices[valid_cutoff:])


class IndiceSplitter(Splitter):
  """
    Class for splits based on input order.
    """

  def __init__(self, verbose=False, valid_indices=None, test_indices=None):
    """
        Parameters
        -----------
        valid_indices: list of int
            indices of samples in the valid set
        test_indices: list of int
            indices of samples in the test set
        """
    self.verbose = verbose
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
        Splits internal compounds into train/validation/test in designated order.
        """
    num_datapoints = len(dataset)
    indices = np.arange(num_datapoints).tolist()
    train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []
    valid_test = self.valid_indices
    valid_test.extend(self.test_indices)
    for indice in indices:
      if not indice in valid_test:
        train_indices.append(indice)

    return (train_indices, self.valid_indices, self.test_indices)


def ClusterFps(fps, cutoff=0.2):
  # (ytz): this is directly copypasta'd from Greg Landrum's clustering example.
  dists = []
  nfps = len(fps)
  for i in range(1, nfps):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - x for x in sims])
  cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
  return cs


class ButinaSplitter(Splitter):
  """
    Class for doing data splits based on the butina clustering of a bulk tanimoto
    fingerprint matrix.
    """

  def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=1000,
            cutoff=0.18):
    """
        Splits internal compounds into train and validation based on the butina
        clustering algorithm. This splitting algorithm has an O(N^2) run time, where N
        is the number of elements in the dataset. The dataset is expected to be a classification
        dataset.

        This algorithm is designed to generate validation data that are novel chemotypes.

        Note that this function entirely disregards the ratios for frac_train, frac_valid,
        and frac_test. Furthermore, it does not generate a test set, only a train and valid set.

        Setting a small cutoff value will generate smaller, finer clusters of high similarity,
        whereas setting a large cutoff value will generate larger, coarser clusters of low similarity.
        """
    print("Performing butina clustering with cutoff of", cutoff)
    mols = []
    for ind, smiles in enumerate(dataset.ids):
      mols.append(Chem.MolFromSmiles(smiles))
    n_mols = len(mols)
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

    scaffold_sets = ClusterFps(fps, cutoff=cutoff)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))

    ys = dataset.y
    valid_inds = []
    for c_idx, cluster in enumerate(scaffold_sets):
      # for m_idx in cluster:
      valid_inds.extend(cluster)
      # continue until we find an active in all the tasks, otherwise we can't
      # compute a meaningful AUC
      # TODO (ytz): really, we want at least one active and inactive in both scenarios.
      # TODO (Ytz): for regression tasks we'd stop after only one cluster.
      active_populations = np.sum(ys[valid_inds], axis=0)
      if np.all(active_populations):
        print("# of actives per task in valid:", active_populations)
        print("Total # of validation points:", len(valid_inds))
        break

    train_inds = list(itertools.chain.from_iterable(scaffold_sets[c_idx + 1:]))
    test_inds = []

    return train_inds, valid_inds, []


class ScaffoldSplitter(Splitter):
  """
    Class for doing data splits based on the scaffold of small molecules.
    """

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
        Splits internal compounds into train/validation/test by scaffold.
        """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    scaffolds = {}
    log("About to generate scaffolds", self.verbose)
    data_len = len(dataset)
    for ind, smiles in enumerate(dataset.ids):
      if ind % log_every_n == 0:
        log("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
      scaffold = generate_scaffold(smiles)
      if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
      else:
        scaffolds[scaffold].append(ind)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_inds, valid_inds, test_inds = [], [], []
    log("About to sort in scaffold sets", self.verbose)
    for scaffold_set in scaffold_sets:
      if len(train_inds) + len(scaffold_set) > train_cutoff:
        if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
          test_inds += scaffold_set
        else:
          valid_inds += scaffold_set
      else:
        train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


class FingerprintSplitter(Splitter):
  """
    Class for doing data splits based on the fingerprints of small molecules
    O(N**2) algorithm
  """

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
        Splits internal compounds into train/validation/test by fingerprint.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    data_len = len(dataset)
    mols, fingerprints = [], []
    train_inds, valid_inds, test_inds = [], [], []
    for ind, smiles in enumerate(dataset.ids):
      mol = Chem.MolFromSmiles(smiles, sanitize=False)
      mols.append(mol)
      fp = FingerprintMols.FingerprintMol(mol)
      fingerprints.append(fp)

    distances = np.ones(shape=(data_len, data_len))
    for i in range(data_len):
      for j in range(data_len):
        distances[i][j] = 1 - DataStructs.FingerprintSimilarity(
            fingerprints[i], fingerprints[j])

    train_cutoff = int(frac_train * len(dataset))
    valid_cutoff = int(frac_valid * len(dataset))

    # Pick the mol closest to everything as the first element of training
    closest_ligand = np.argmin(np.sum(distances, axis=1))
    train_inds.append(closest_ligand)
    cur_distances = [float('inf')] * data_len
    self.update_distances(closest_ligand, cur_distances, distances, train_inds)
    for i in range(1, train_cutoff):
      closest_ligand = np.argmin(cur_distances)
      train_inds.append(closest_ligand)
      self.update_distances(closest_ligand, cur_distances, distances,
                            train_inds)

    # Pick the closest mol from what is left
    index, best_dist = 0, float('inf')
    for i in range(data_len):
      if i in train_inds:
        continue
      dist = np.sum(distances[i])
      if dist < best_dist:
        index, best_dist = i, dist
    valid_inds.append(index)

    leave_out_indexes = train_inds + valid_inds
    cur_distances = [float('inf')] * data_len
    self.update_distances(index, cur_distances, distances, leave_out_indexes)
    for i in range(1, valid_cutoff):
      closest_ligand = np.argmin(cur_distances)
      valid_inds.append(closest_ligand)
      leave_out_indexes.append(closest_ligand)
      self.update_distances(closest_ligand, cur_distances, distances,
                            leave_out_indexes)

    # Test is everything else
    for i in range(data_len):
      if i in leave_out_indexes:
        continue
      test_inds.append(i)
    return train_inds, valid_inds, test_inds

  def update_distances(self, last_selected, cur_distances, distance_matrix,
                       dont_update):
    for i in range(len(cur_distances)):
      if i in dont_update:
        cur_distances[i] = float('inf')
        continue
      new_dist = distance_matrix[i][last_selected]
      if new_dist < cur_distances[i]:
        cur_distances[i] = new_dist


class SpecifiedSplitter(Splitter):
  """
    Class that splits data according to user specification.
    """

  def __init__(self, input_file, split_field, verbose=False):
    """Provide input information for splits."""
    raw_df = next(load_data([input_file], shard_size=None))
    self.splits = raw_df[split_field].values
    self.verbose = verbose

  def split(self,
            dataset,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
        Splits internal compounds into train/validation/test by user-specification.
        """
    train_inds, valid_inds, test_inds = [], [], []
    for ind, split in enumerate(self.splits):
      split = split.lower()
      if split == "train":
        train_inds.append(ind)
      elif split in ["valid", "validation"]:
        valid_inds.append(ind)
      elif split == "test":
        test_inds.append(ind)
      else:
        raise ValueError("Missing required split information.")
    return train_inds, valid_inds, test_inds


class TimeSplitterPDBbind(Splitter):

  def __init__(self, ids, year_file=None, verbose=False):
    self.ids = ids
    self.year_file = year_file
    self.verbose = verbose

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits protein-ligand pairs in PDBbind into train/validation/test in time order.
    """
    if self.year_file is None:
      try:
        data_dir = os.environ['DEEPCHEM_DATA_DIR']
        self.year_file = os.path.join(data_dir, 'pdbbind_year.csv')
        if not os.path.exists(self.year_file):
          dc.utils.download_url(
              'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_year.csv',
              dest_dir=data_dir)
      except:
        raise ValueError("Time description file should be specified")
    df = pd.read_csv(self.year_file, header=None)
    self.years = {}
    for i in range(df.shape[0]):
      self.years[df[0][i]] = int(df[1][i])
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    assert len(self.ids) == num_datapoints
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = range(num_datapoints)
    data_year = [self.years[self.ids[i]] for i in indices]
    new_indices = [
        pair[0] for pair in sorted(zip(indices, data_year), key=lambda x: x[1])
    ]

    return (new_indices[:train_cutoff], new_indices[train_cutoff:valid_cutoff],
            new_indices[valid_cutoff:])
