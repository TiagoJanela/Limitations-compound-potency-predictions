from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from regression_training_size.ML.ml_utils_reg import potency_classes


def get_uniformly_distributed_sample_idx(data: Union[np.ndarray, List[float]], cid: Union[np.ndarray, List[str]],
                                         n_sample: int, bins: Union[int, List[float]], seed: int, verbose: bool = False,
                                         idx: bool = True) -> Union[List[int], List[str]]:

    """     Description:
    This function is designed to obtain a uniformly distributed sample of data from a given dataset.
    The function returns either the indices or the corresponding cids of the selected data points.

    Parameters:
    data: 1D array or list of data points to sample from (potency)
    cid: 1D array or list of IDs for the corresponding data points
    n_sample: number of data points to sample
    bins: boundaries of the bins to group the data into
    seed: random seed for reproducibility
        verbose (optional): if set to True, the function will print out additional information during its execution
    idx (optional): if set to True, the function will return the indices of the selected data points.
    If set to False, the function will return the corresponding cids.

    Returns:
    If idx=True: a sorted list of indices corresponding to the selected data points
    If idx=False: a list of cids corresponding to the selected data points"""

    df = pd.DataFrame({'value': data, 'cid': cid}).reset_index().rename({'index': 'ID'}, axis=1)
    assert df.shape[0] >= n_sample
    df['bin'] = df['value'].apply(lambda x: np.digitize(x, bins))

    n_bin_sample = int(np.floor(n_sample / df['bin'].nunique()))

    n_missing = n_sample
    df_sample = pd.DataFrame()
    df_remaining = df
    n_available_bins = df['bin'].nunique()
    while n_missing >= n_available_bins:
        if verbose:
            print(f"Available pool size: {df_remaining.shape[0]}")
            print(f"Number of datapoints to select per bin: {n_bin_sample}")

        sizes_bins = df_remaining.groupby('bin').size()

        if verbose:
            print("Bin sizes:", sizes_bins.to_dict())

        bins_enough_data = sizes_bins[sizes_bins >= n_bin_sample].index.to_list()

        if verbose:
            print("Bins with enough data", bins_enough_data)

        df_sample_enough = pd.DataFrame()
        if len(bins_enough_data) != 0:
            df_sample_enough = df_remaining[df_remaining['bin'].isin(bins_enough_data)].groupby(
                'bin').sample(n_bin_sample, random_state=seed)
        df_sample_not_enough = df_remaining[~df_remaining['bin'].isin(bins_enough_data)]
        df_sample = pd.concat([df_sample, df_sample_enough, df_sample_not_enough])
        del df_sample_enough

        if verbose:
            print(f"Sample size: {df_sample.shape[0]}")

        n_missing = n_sample - df_sample.shape[0]

        df_remaining = df_remaining[~df_remaining['ID'].isin(df_sample['ID'].to_numpy())]

        if df_remaining['bin'].nunique() > 0:
            n_bin_sample = int(np.floor(n_missing / df_remaining['bin'].nunique()))

        if verbose:
            print(f"Missing samples: {n_missing}\n")

    if n_missing > 0:
        if verbose:
            print(f"Select {n_missing} samples randomly")

        df_sample = pd.concat([df_sample, df_remaining.sample(n_missing, random_state=seed)])

    if verbose:
        print(f"Selected {df_sample.shape[0]} samples")
    assert df_sample.shape[0] == n_sample, f"Something went wrong."

    if idx:
        return sorted(df_sample['ID'].to_list())
    else:
        return df_sample['cid'].tolist()

def select_train_subsets(df_train: pd.DataFrame, sizes: List[int], seed: int, pot_bins: Union[int, List[float]]) -> Dict[int, List[str]]:

    """This function takes a pandas DataFrame df_train, list of integers sizes, integer seed and a Union of integer
    pot_bins or list of floats pot_bins as input. It returns a dictionary of selected chemical compound IDs from
    the df_train DataFrame based on the size criteria mentioned in sizes input and pot_bins which specifies the number
    of bins to divide the potential (pPot) values in df_train DataFrame. The returned dictionary contains keys as the
    size values mentioned in the sizes input, and the values are the corresponding selected chemical compound IDs.

    Input arguments:

    df_train (pandas DataFrame): The input pandas DataFrame containing the potential (pPot) values of chemical compounds
    to be selected from.

    sizes (list of integers): A list of integers specifying the size criteria for selecting chemical compounds from
    the input df_train DataFrame.

    seed (integer): An integer used as a seed for random sampling.

    pot_bins (Union[int, List[float]]): The number of bins to divide the potential (pPot) values in df_train DataFrame.
                                        If pot_bins is an integer, it is used as the number of bins, and if it is a list
                                        of floats, the values in the list are used as bin edges.

    Output:

    total_selection (dictionary): A dictionary containing the selected chemical compound IDs from the df_train DataFrame
    based on the size criteria mentioned in the sizes input and pot_bins. The keys of the dictionary are the size values
    mentioned in the sizes input, and the values are the corresponding selected chemical compound IDs."""

    total_selection = {}

    last_size = 0
    last_added_cpds = []

    for size in sizes:

        df_train = df_train[~df_train['chembl_cid'].isin(set([item for sublist in list(total_selection.values()) for item in sublist]))]

        n_select = size - last_size

        current_selection = {size: last_added_cpds + get_uniformly_distributed_sample_idx(df_train.pPot.values,
                                                                                          df_train.chembl_cid.values,
                                                                                          n_select, bins=pot_bins,
                                                                                          seed=seed, verbose=False,
                                                                                          idx=False)}

        total_selection = {**total_selection, **current_selection}
        last_size = size
        last_added_cpds = list(current_selection.values())[0]

    return total_selection


def dataset_train_test(df_regression: pd.DataFrame, pot_bins: Optional[List[int]] = None, tr_set_sizes: Optional[List[int]] = None,
                       balance_test_set: bool = True, n_trials: int = 10, plot: bool = False, shuffle=False) -> pd.DataFrame:

    """ This function splits the input dataset into multiple training and testing sets, and returns a dataframe containing
    the split selections. For each trial, the function first selects training set sizes specified in `tr_set_sizes` from
    the input dataset using `select_train_subsets()` function. Then, it selects test set samples, balancing the classes
    of samples in the test set if `balance_test_set` is True, and using a 50/50 split if `balance_test_set` is False.

    Parameters:
    -----------
    - df_regression: pandas.DataFrame
    Input dataset to be split into train and test sets.
    - pot_bins: list or None (default: None)
    A list of potency class bin labels to be used for splitting the dataset. Default value is None, which will use
    the labels `[5, 7, 9]` as the default potency class bin labels.
    - tr_set_sizes: list or None (default: None)
    A list of training set sizes to be selected from the input dataset using `select_train_subsets()` function.
    Default value is None, which will use the list `[100, 300, 500, 1000, 2000, 5000]` as the default training
    set sizes.
    - balance_test_set: bool (default: True)
    Whether or not to balance the test set by sampling the same number of compounds from each potency class bin.
    If True, the smallest potency class bin is used to determine the number of test set samples to be selected.
    If False, a 50/50 split is used for selecting test set samples.
    - n_trials: int (default: 10)
    Number of trials to perform for splitting the dataset into training and test sets.
    - plot: bool (default: False)
    Whether or not to plot the distribution of the selected train and test sets.

    Returns:
    --------
    A pandas.DataFrame containing the following columns:
    - chembl_cid: str
    ChEMBL compound ID
    - pPot: float
    Potency value of the compound
    - potency_class: int
    Potency class bin label of the compound
    - Target ID: str
    Target identifier for the compound
    - dataset: str
    The dataset that the compound belongs to (`train` or `test`)
    - trial: int
    The trial number of the split
    - set_size: int
    The size of the train or test set """

    if pot_bins is None:
        pot_bins = [5, 7, 9]

    df_selections = pd.DataFrame()

    for i, trial in enumerate(np.arange(0, n_trials, 1)):

        df_cpds = df_regression
        n_needed_cpds = int(tr_set_sizes[-1])
        df_train_pool_idx = get_uniformly_distributed_sample_idx(df_cpds.pPot.values, df_cpds.chembl_cid.values,
                                                                 n_needed_cpds, bins=pot_bins, seed=trial,
                                                                 verbose=False)

        df_train_pool = df_cpds.iloc[df_train_pool_idx]

        if plot:
            #print('train_pool')
            sns.histplot(df_train_pool, x='pPot', bins=[5,7,9,11])
            plt.show()
            plt.clf()

        # remove used CPDs (in training)
        df_cpds = df_cpds[~df_cpds['chembl_cid'].isin(df_train_pool['chembl_cid'])]

        # select test data
        if balance_test_set:
            smallest_pot_class_cpds = min(dict(df_cpds['potency_class'].value_counts()).values())
            n_test = smallest_pot_class_cpds*len(pot_bins)
            test_idx = get_uniformly_distributed_sample_idx(df_cpds.pPot.values, df_cpds.chembl_cid.values, n_test,
                                                            bins=pot_bins, seed=trial, verbose=False)
            df_test = df_cpds.iloc[test_idx]
            if shuffle:
                np.random.seed(i)
                df_test['pPot_shuffle'] = np.random.permutation(df_test.pPot)
                df_test['potency_class_shuffle'] = potency_classes(df_test.pPot_shuffle.values, [5, 7, 9, 11])
                assert len(df_test.loc[df_test.potency_class_shuffle == pot_bins[0]]) == len(df_test.loc[df_test.potency_class_shuffle == pot_bins[1]]) == len(df_test.loc[df_test.potency_class_shuffle == pot_bins[2]])
            else:
                assert len(df_test.loc[df_test.potency_class == pot_bins[0]]) == len(df_test.loc[df_test.potency_class == pot_bins[1]]) == len(df_test.loc[df_test.potency_class == pot_bins[2]])

        else:
            df_test = df_cpds

            if shuffle:
                np.random.seed(i)
                df_test['pPot_shuffle'] = np.random.permutation(df_test.pPot)
                df_test['potency_class_shuffle'] = potency_classes(df_test.pPot_shuffle.values, [5, 7, 9, 11])

        target = df_cpds.chembl_tid.unique().tolist()[0]

        df_test['Target ID'] = target
        df_test['dataset'] = 'test'
        df_test['trial'] = trial
        df_test['set_size'] = len(df_test)
        df_selections = df_selections.append(df_test)

        if plot:
            print('test')
            sns.histplot(df_test, x='pPot', bins=[5,7,9,11])
            plt.show()
            plt.clf()

        df_train_dict = select_train_subsets(df_train_pool, tr_set_sizes, trial, pot_bins)

        if not shuffle:
            for size in tr_set_sizes:

                df_train = df_train_pool[df_train_pool['chembl_cid'].isin(df_train_dict.get(size))]
                df_train['Target ID'] = target
                df_train['dataset'] = 'train'
                df_train['trial'] = trial
                df_train['set_size'] = len(df_train)
                df_selections = df_selections.append(df_train)

                if plot:
                    print('train', size)
                    sns.histplot(df_train, x='pPot', bins=[5,7,9,11])
                    plt.show()
                    plt.clf()
        else:
            df_train_cur = None
            np.random.seed(i)
            for size in tr_set_sizes:
                if df_train_cur is None:
                    df_train = df_train_pool[df_train_pool['chembl_cid'].isin(df_train_dict.get(size))]
                    df_train['pPot_shuffle'] = np.random.permutation(df_train.pPot)
                    df_train['potency_class_shuffle'] = potency_classes(df_train.pPot_shuffle.values, [5, 7, 9, 11])
                    df_train['Target ID'] = target
                    df_train['dataset'] = 'train'
                    df_train['trial'] = trial
                    df_train['set_size'] = len(df_train)
                    df_train_cur = df_train
                    df_selections = df_selections.append(df_train)
                else:
                    train_ids = set(df_train_dict.get(size)) ^ set(df_train_cur.chembl_cid.tolist())
                    df_train_new = df_train_pool[df_train_pool['chembl_cid'].isin(train_ids)]
                    df_train_new['pPot_shuffle'] = np.random.permutation(df_train_new.pPot)
                    df_train_new['potency_class_shuffle'] = potency_classes(df_train_new.pPot_shuffle.values, [5, 7, 9, 11])

                    df_train = pd.concat([df_train_cur, df_train_new])
                    df_train['Target ID'] = target
                    df_train['dataset'] = 'train'
                    df_train['trial'] = trial
                    df_train['set_size'] = len(df_train)
                    df_train_cur = df_train
                    df_selections = df_selections.append(df_train)

    return df_selections