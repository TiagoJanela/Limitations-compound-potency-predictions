# imports
import itertools as it
import os
import random
from collections import OrderedDict
from itertools import tee
from typing import List, Dict

# Plotting
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
# Rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from scipy.stats import stats
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TanimotoKernel:
    def __init__(self, sparse_features=False):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


def tanimoto_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_dense(matrix_a, matrix_b)


def maxminpicker(fp_list, ntopick, seed=None):
    from rdkit import SimDivFilters
    mmp = SimDivFilters.MaxMinPicker()
    # n_to_pick = round(ntopick * len(fp_list))
    picks = mmp.LazyBitVectorPick(fp_list, len(fp_list), ntopick, seed=seed)
    return list(picks)


def create_directory(path: str, verbose: bool = True):
    if not os.path.exists(path):

        if len(path.split("/")) <= 2:
            os.mkdir(path)
        else:
            os.makedirs(path)
        if verbose:
            print(f"Created new directory '{path}'")
    return path


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj


def ECFP4(smiles_list: List[str], n_bits=2048, radius=2) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Converts array of SMILES to ECFP bitvectors.
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
        n_bits: number of bits
        radius: ECFP fingerprint radius

    Returns: RDKit mol objects [List]
    """
    mols = construct_check_mol_list(smiles_list)
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, n_bits) for m in mols]


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def potency_classes(potency_values: list, potency_bins=None):
    if potency_bins is None:
        potency_bins = [5, 6, 7, 8, 9, 10, 11]

    pot_bin = []
    for pot in potency_values:
        pot_idx = pairwise(potency_bins)
        for idx in list(pot_idx):
            if idx[1] == 11:
                if idx[0] <= pot <= idx[1]:
                    pot_bin.append(idx[0])

            elif idx[0] <= pot < idx[1]:
                pot_bin.append(idx[0])

    return pot_bin


def get_uniformly_distributed_sample(data, n_sample, bins, seed, verbose: bool = False):
    df = pd.DataFrame({'value': data}).reset_index().rename({'index': 'ID'}, axis=1)
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
    return sorted(df_sample['ID'].to_list())


# Plot Boxplot
def plot_regression(df, metric, plot_type='boxplot',
                    filename=None, results_path=None,
                    x="Target ID",
                    ymin=0.2, ymax=None, yticks=None,
                    palette='tab10',
                    hue=None, hue_order=None,
                    legend=True,
                    font_size=18, fig_size_x=16, fig_size_y=12,
                    bbox_to_anchor=(.5, -0.18)):
    performance_df_ = df.loc[df['Metric'] == f'{metric}']

    font = {'size': font_size}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(figsize=(fig_size_x, fig_size_y))
    if plot_type == 'boxplot':
        sns.boxplot(data=performance_df_, x=x, y="Value", palette=palette, hue=hue, hue_order=hue_order)
    elif plot_type == 'barplot':
        sns.barplot(data=performance_df_, x=x, y="Value", palette=palette, hue=hue, hue_order=hue_order)
    plt.tight_layout()
    if yticks is not None:
        axs.set(ylim=(ymin, ymax), yticks=yticks)
    if legend:
        plt.legend(loc='lower center', prop={'size': font_size}, bbox_to_anchor=bbox_to_anchor, ncol=6, frameon=False)
    plt.xlabel(' ')
    plt.ylabel(f'{metric}', labelpad=10)
    plt.xlabel('Target ID', labelpad=10)
    plt.style.use('classic')
    if results_path:
        plt.savefig(results_path + f'{filename}_{metric}.png', dpi=300, bbox_inches="tight")
    plt.show()


def scaffold_split(data, seed=42, test_size=0.2, n_splits=10, n_cpds_tolerance=5):
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for idx, core in enumerate(data.core):
        scaffolds[core].append(idx)

    n_total_test = int(np.floor(test_size * len(data)))
    rng = np.random.RandomState(seed)
    for i in range(n_splits):

        scaffold_sets = rng.permutation(list(scaffolds.values()))
        scaffold_sets = np.array(scaffold_sets, dtype=object)

        train_index = []
        test_index = []

        for scaffold_set in scaffold_sets:
            if len(test_index) + len(scaffold_set) <= n_total_test:
                test_index.extend(scaffold_set)
            else:
                train_index.extend(scaffold_set)
        assert np.abs(len(test_index) - n_total_test) <= n_cpds_tolerance, f'There are {len(test_index)} CPDs in the ' \
                                                                           f'test set, but {n_total_test} are expected '
        yield train_index, test_index


def plot_regression_models_cat(df, metric, plot_type='boxplot',
                               filename=None, results_path=None,
                               x="Algorithm", y='Value',
                               col="Metric",
                               row=None,
                               ymin=None, ymax=None, yticks=None,
                               xticks=None,
                               palette='tab10',
                               x_labels='',
                               y_labels='',
                               hue=None, hue_order=None, title=True,
                               order=None,
                               col_nr=2,
                               font_size=18, height=10, aspect=1.2, width=None, bbox_to_anchor=(-0.01, -0.15),
                               sharey=True,
                               legend_title=None, sub_fig_title=None, **kwargs):
    # database
    performance_df_ = df.loc[df.Metric.isin(metric)]

    # plt parameters
    font = {'size': font_size}
    matplotlib.rc('font', **font)

    if plot_type == 'boxplot':
        kind = "box"
    elif plot_type == 'barplot':
        kind = "bar"
    elif plot_type == 'pointplot':
        kind = "point"

    g = sns.catplot(data=performance_df_, x=x, y=y, col=col,
                    kind=kind,
                    height=height, aspect=aspect,  # width=width,
                    order=order, palette=palette,
                    hue=hue,
                    hue_order=hue_order,  # col_wrap=col_nr,
                    row=row,
                    legend=False, sharey=sharey, **kwargs)

    g.set_ylabels(y_labels, labelpad=10)
    if row and len(metric) > 1:
        g.axes[0, 0].set_ylabel(f'{metric[0]}', labelpad=10)
        g.axes[1, 0].set_ylabel(f'{metric[1]}', labelpad=10)
    elif col and len(metric) > 1:
        g.axes[0, 0].set_ylabel(f'{metric[0]}', labelpad=10)
        g.axes[0, 1].set_ylabel(f'{metric[1]}', labelpad=10)
    else:
        g.set_ylabels(y_labels, labelpad=10)

    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize=60, x=0, y=0.95, fontweight='bold')

    tids = sorted(performance_df_['Target ID'].unique())
    g.set_xlabels(f'{x_labels}', labelpad=10)
    g.set(ylim=(ymin, ymax))
    if title:
        if row:
            g.set_titles(r"{row_var}: {row_name} - {col_var}: {col_name}", )  # fontweight="bold"

        if isinstance(title, str):
            g.set_titles(title)
        else:
            if row is None:
                g.set_titles("{col_var}: {col_name}")
            else:
                # g.set_titles("{row_var}: {row_name} - ({col_name})")
                g.set_titles("{col_var}: {col_name} - ({row_name})")
                # g.set_titles("{col_var}: {col_name}")

    if yticks:
        g.set(ylim=(ymin, ymax), yticks=yticks)
    if xticks:
        g.set_xticklabels(xticks)
    plt.tight_layout()
    g.despine(right=True, top=True)
    plt.legend(loc='lower center', prop={'size': font_size}, bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order),
               frameon=False, title=legend_title, labelspacing=2)

    if results_path:
        plt.savefig(results_path + f'{filename}.png', dpi=300, bbox_inches="tight")
    plt.show()


def plot_regression_models_cat_mod(df, metric, plot_type='boxplot',
                                   filename=None, results_path=None,
                                   x="Algorithm", y='Value',
                                   col="Metric",
                                   row=None,
                                   ymin=None, ymax=None, yticks=None,
                                   xticks=None,
                                   palette='tab10',
                                   x_labels='',
                                   y_labels='',
                                   hue=None, hue_order=None, title=True,
                                   order=None,
                                   col_nr=2,
                                   font_size=18, height=10, aspect=1.2, width=None, bbox_to_anchor=(-0.01, -0.15),
                                   sharey=True,
                                   legend_title=None, sub_fig_title=None, **kwargs):
    # database
    performance_df_ = df.loc[df.Metric.isin(metric)]

    # plt parameters
    font = {'size': font_size}
    matplotlib.rc('font', **font)

    if plot_type == 'boxplot':
        kind = "box"
    elif plot_type == 'barplot':
        kind = "bar"
    elif plot_type == 'pointplot':
        kind = "point"

    g = sns.catplot(data=performance_df_, x=x, y=y, col=col,
                    kind=kind,
                    height=height, aspect=aspect,  # width=width,
                    order=order, palette=palette,
                    hue=hue,
                    hue_order=hue_order,  # col_wrap=col_nr,
                    row=row,
                    legend=False, sharey=sharey, **kwargs)

    #for axis in g.axes.flat:
    #    axis.tick_params(labelright=False)
    #    if axis.is_first_col():
    #        axis.set_ylabel(f'{metric[0]}', labelpad=10)
    #    else:
    #        axis.set_ylabel(f'{metric[1]}', labelpad=10)

    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize=20, x=0, y=0, fontweight='bold')

    tids = sorted(performance_df_['Target ID'].unique())
    g.set_xlabels(f'{x_labels}', labelpad=10)
    g.set_ylabels(y_labels, labelpad=10)
    g.set(ylim=(ymin, ymax))
    if title:
        if row:
            g.set_titles(r"{row_name} - {col_name}", )  # fontweight="bold"

        if isinstance(title, str):
            g.set_titles(title)
        else:
            if row is None:
                g.set_titles("{col_var}: {col_name}")
            #else:
            #    for ax in g.axes.flat:
            #        if ax.is_first_row():
             #           ax.set_title(f"Target ID: {tids[0]}")
             #       elif ax.is_last_row():
             #           ax.set_title(f"{tids[2]}")
             #       else:
              #          ax.set_title(f"{tids[1]}")
    if yticks:
        g.set(ylim=(ymin, ymax), yticks=yticks)
    if xticks:
        g.set_xticklabels(xticks)
    plt.tight_layout()
    g.despine(right=True, top=True)
    plt.legend(loc='lower center', prop={'size': font_size}, bbox_to_anchor=bbox_to_anchor, ncol=len(hue_order),
               frameon=False, title=legend_title, labelspacing=2)

    if results_path:
        plt.savefig(results_path + f'{filename}.png', dpi=300, bbox_inches="tight")
    plt.show()


def select_train_subsets_unbalanced(df_train: pd.DataFrame, sizes: List[int], seed: int, ) -> Dict[int, List[str]]:
    """
    Selects subsets of the training set for the unbalanced case
    :param df_train: dataframe with the training set
    :param sizes: sizes of the subsets
    :param seed: seed for reproducibility
    :return: dictionary with the selected subsets containing the cpd ids
    """

    total_selection = {}

    last_size = 0
    last_added_cpds = []

    for size in sizes:
        df_train = df_train[
            ~df_train['chembl_cid'].isin(set([item for sublist in list(total_selection.values()) for item in sublist]))]

        n_select = size - last_size

        cpd_selection = last_added_cpds + df_train.sample(n_select, random_state=seed).chembl_cid.tolist()

        assert len(cpd_selection) == len(set(cpd_selection))

        current_selection = {size: cpd_selection}

        total_selection = {**total_selection, **current_selection}
        last_size = size
        last_added_cpds = list(current_selection.values())[0]

    return total_selection


def metrics_potency_classes(df, targets=[280, 203, 2409],
                            training_sizes=[6, 12, 18, 30, 48, 78, 126, 204, 330],
                            pot_classes=[5, 7, 9],
                            algorithms=['1-NN', 'kNN', 'SVR', 'RFR', 'MR'],
                            trials=10):
    """
    Calculates the metrics for the potency classes
    :param df: Dataframe with the results
    :return: Metrics for the potency classes
    """
    db_query = OrderedDict({'target': targets,
                            'tr_sizes': training_sizes,
                            'pot_class': pot_classes,
                            'trial': [i for i in range(trials)],
                            'algorithm': algorithms})

    pot_classes_performance = []
    db_search_query = {n: {name: value for name, value in zip(db_query.keys(), comb)} for n, comb in
                       enumerate(it.product(*list(db_query.values())), 1)}

    for i, idx_params in enumerate(db_search_query):
        cur_params = db_search_query[idx_params]
        pot_df = df

        pot_trial = pot_df.loc[(pot_df['Target ID'] == cur_params.get('target')) &
                               (pot_df['Training size'] == cur_params.get('tr_sizes')) &
                               (pot_df['potency_class'] == cur_params.get('pot_class')) &
                               (pot_df['trial'] == cur_params.get('trial')) &
                               (pot_df['Algorithm'] == cur_params.get('algorithm'))
                               ]

        result_dict = {"Target ID": cur_params.get('target'),
                       "trial": cur_params.get('trial'),
                       "Algorithm": cur_params.get('algorithm'),
                       "potency_class": cur_params.get('pot_class'),
                       "Training size": int(cur_params.get('tr_sizes')),
                       "Test size": len(pot_trial),

                       "MAE": mean_absolute_error(pot_trial['Experimental'], pot_trial['Predicted']),
                       "MSE": mean_squared_error(pot_trial['Experimental'], pot_trial['Predicted']),
                       "RMSE": mean_squared_error(pot_trial['Experimental'], pot_trial['Predicted'],
                                                  squared=False),
                       "R2": metrics.r2_score(pot_trial['Experimental'], pot_trial['Predicted']),
                       "Pearsonr": stats.pearsonr(pot_trial['Experimental'], pot_trial['Predicted'])[0],
                       "r²": stats.pearsonr(pot_trial['Experimental'], pot_trial['Predicted'])[0]**2,
                       "Spearmanr": stats.spearmanr(pot_trial['Experimental'], pot_trial['Predicted'])[0]
                       }

        pot_classes_performance.append(result_dict)
    potency_class_df = pd.DataFrame(pot_classes_performance)
    results_pc = potency_class_df[
        ["Target ID", "Algorithm", "Test size", "potency_class", "trial", "Training size", "MAE", "MSE", "RMSE", "R2", "r²",
         "Pearsonr", "Spearmanr"]]
    results_pc.set_index(["Target ID", "Algorithm", "Test size", "potency_class", "trial", "Training size"],
                         inplace=True)
    results_pc.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2", "r²", "Pearsonr", "Spearmanr"]],
                                                    names=["Value", "Metric"])
    results_pc = results_pc.stack().reset_index().set_index("Target ID")
    results_pc.reset_index(inplace=True)
    return results_pc


def metric_potency_classes_ub(pred_df,
                              targets,
                              potency_classess,
                              algorithms,
                              trials):

    db_query = OrderedDict({'target': targets,
                            'pot_class': potency_classess,
                            'trial': [i for i in range(trials)],
                            'algorithm': algorithms})

    pot_classes_performance = []
    db_search_query = {n: {name: value for name, value in zip(db_query.keys(), comb)} for n, comb in
                       enumerate(it.product(*list(db_query.values())), 1)}
    for i, idx_params in enumerate(db_search_query):
        cur_params = db_search_query[idx_params]
        pot_df = pred_df.copy()

        pot_trial = pot_df.loc[(pot_df['Target ID'] == cur_params.get('target')) &
                               (pot_df['potency_class'] == cur_params.get('pot_class')) &
                               (pot_df['trial'] == cur_params.get('trial')) &
                               (pot_df['Algorithm'] == cur_params.get('algorithm'))
                               ]

        result_dict = {"Target ID": cur_params.get('target'),
                       "trial": cur_params.get('trial'),
                       "Algorithm": cur_params.get('algorithm'),
                       "potency_class": cur_params.get('pot_class'),
                       "Test size": len(pot_trial),
                       "MAE": mean_absolute_error(pot_trial['Experimental'], pot_trial['Predicted']),
                       "MSE": mean_squared_error(pot_trial['Experimental'], pot_trial['Predicted']),
                       "RMSE": mean_squared_error(pot_trial['Experimental'], pot_trial['Predicted'],
                                                  squared=False),
                       "R2": metrics.r2_score(pot_trial['Experimental'], pot_trial['Predicted']),
                       "r²": stats.pearsonr(pot_trial['Experimental'], pot_trial['Predicted'])[0]**2,
                       }

        pot_classes_performance.append(result_dict)
    potency_class_df = pd.DataFrame(pot_classes_performance)
    results_pc = potency_class_df[
        ["Target ID", "Algorithm", "Test size", "potency_class", "trial", "MAE", "MSE", "RMSE", "R2", "r²"]]
    results_pc.set_index(["Target ID", "Algorithm", "Test size", "potency_class", "trial"], inplace=True)
    results_pc.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2", "r²"]],
                                                    names=["Value", "Metric"])
    results_pc = results_pc.stack().reset_index().set_index("Target ID")
    results_pc.reset_index(inplace=True)
    results_pc.potency_class.replace({5: '5 - 7', 7: '7 - 9', 9: '9 - 11'}, inplace=True)

    return results_pc


def plot_heatmap_stat_analysis(df, x, y, value, pvalue_boundaries = [0, 0.005, 1],
                               font_size=16, clrs = ['green', 'red'],
                               height=10, aspect=1.5, square=False, results_path = None, filename=None,
                               sub_fig_title=None, **kwargs):

    norm=matplotlib.colors.BoundaryNorm(pvalue_boundaries, ncolors=len(pvalue_boundaries))
    cmap=matplotlib.colors.ListedColormap(clrs)

    pvalue_boundaries = [0, 0.005, 1]
    font = {'size': font_size}
    matplotlib.rc('font', **font)

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, **kwargs)

    ax = sns.FacetGrid(df, height=height, aspect=aspect, **kwargs).map_dataframe(draw_heatmap, x, y, value,
                                                                                 norm=norm, cmap=cmap, cbar=False,
                                                                                 annot=True, square=square)

    ax.set_titles("{col_var}: {col_name}")
    #ax.set_xticklabels(rotation=0)

    fig = ax.figure
    cbar_ax = fig.add_axes([1.01, 0.36, 0.02, 0.3])
    cbar_ticks = np.linspace(0, 1, (len(pvalue_boundaries) - 1) * 2 + 1)[1:][::2]

    cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap.reversed(), ticks=cbar_ticks)

    cbar.set_ticks(ticks=cbar_ticks,
                   labels=[f'p < {p}' if p < 1 else 'Not significant' for p in pvalue_boundaries[1:]][::-1])

    cbar.outline.set_edgecolor('0.5')
    cbar.outline.set_linewidth(0.1)
    cbar.ax.tick_params(size=0)
    if sub_fig_title:
        plt.suptitle(f'{sub_fig_title}', fontsize = 25, x=0, y=1, fontweight='bold')
    if filename:
        fig.savefig(f'{results_path}/{filename}.png', dpi=300, bbox_inches='tight')
    return ax