import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy import stats
from statsmodels.formula.api import ols
import argparse
import os
import glob
import warnings
import config_utils

from functools import reduce

import random


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# Which params were sampled logarithmically?
plot_logarithmic = ['learning_rate', 'weight_decay']


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', type=str, help='directory containing log files', default=None)
parser.add_argument('-c', '--csv', type=str, help='csv file from previously read log files', default=None)
parser.add_argument('-i', '--id', type=str, help='Identifier for this analysis and subdirectory to output results to.', default='temp')
parser.add_argument('-r', '--recursive', action='store_true', help='to also read any log files in sub*directories of --log')
parser.add_argument('-p', '--showplots', action='store_true', help='to show each plot prior to saving and continuing')
parser.add_argument('-t', '--histories', action='store_true', help='to plot loss histories (warning: requires slow rereading of log files).')
parser.add_argument('-f', '--full_samples', action='store_true', help='to compute results for all runs of each group; otherwise subsample for fair comparison.')
parser.add_argument('-s', '--score', default='macro_f1', help='validation score of interest', choices=['acc', 'macro_f1', 'macro_f1_main'])

maxima_of_interest = ['val_loss', 'val_acc', 'val_macro_f1', 'val_macro_f1_main', 'train_acc', 'train_macro_f1', 'train_macro_f1_main']

groupings_of_interest = [
                         ['static', 'dynamic'],
                         ['static', 'weights', 'noweights'],
                         ['weights', 'noweights'],
                         ['valueweights', 'valuezeros'],
                         ['dot', 'cos', 'mlp'],
                         ['softmax', 'nosoftmax'],
                          ['keys', 'nokeys'],
                          ['noweights_nokeys', 'weights_nokeys'],
                         ]     # Which main categories to compare

hyperparams_of_interest = {
                           # 'all': [
                           #       'entity_library',
                           #       'gate_type',
                           #       'entlib_weights',
                           #       'gate_nonlinearity',
                           #       'gate_sum_keys_values',
                           #       'gate_softmax',
                           #       'entlib_normalization',
                           # ],
                           # 'noweights':  [
                           #             'gate_type',
                           #             'gate_nonlinearity',
                           #             'gate_softmax',
                           #              'entlib_normalization',
                           # ],
                           # 'noweights_nokeys':  [
                           #             'gate_type',
                           #             'gate_nonlinearity',
                           #             'gate_softmax',
                           #              'entlib_normalization',
                           # ],
                           # 'weights_nokeys':  [
                           #             'gate_type',
                           #             'gate_nonlinearity',
                           #             'gate_softmax',
                           #              'entlib_normalization',
                           # ],
                           # 'dynamic':  [
                           #             'entlib_dim',
                           #             'gate_type',
                           #             'entlib_weights',
                           #             'gate_nonlinearity',
                           #             'gate_sum_keys_values',
                           #             'gate_softmax',
                           #             'entlib_normalization',
                           # ],
                            # 'static': [
                            #     'gate_type',
                            #     'gate_nonlinearity',
                            #     'gate_softmax',
                            # ],
                           }

num_best_to_plot_individually = 5
num_best_to_plot_jointly_per_value = 30

def _main():
    global groupings_of_interest
    global hyperparams_of_interest

    args = parser.parse_args()
    score = 'val_' + args.score

    out_path = os.path.join(args.logdir or os.path.dirname(args.csv), args.id)
    os.makedirs(out_path, exist_ok=True)
    print('Results of analysis will be output to {0}.'.format(out_path))

    # Read new results from log directory if given, and write to csv (else read straight from csv):
    if args.logdir is not None:
        # TODO If csv file already exist, ask for confirmation first to regenerate it (takes long!).

        print('Log directory given.')
        log_dir = args.logdir
        logs = read_logs(log_dir, recursive=args.recursive)

        if len(logs) == 0:
            print("No complete logs found. Quitting.")
            quit()

        args.csv = os.path.join(log_dir, 'scores.csv')
        write_logs_to_csv(logs, args.csv)

    # Read results from csv into dafaframe and organize by group
    df = read_csv(args.csv)
    dfs = data_groups(df)       # dictionary of dataframes

    # Remove empty groups from consideration
    empty_groups = [group for grouping in groupings_of_interest for group in grouping if len(dfs[group]) == 0]
    empty_groups.extend([key for key in hyperparams_of_interest.keys() if len(dfs[key]) == 0])
    empty_groups = list(set(empty_groups))
    groupings_of_interest = [[x for x in grouping if x not in empty_groups] for grouping in groupings_of_interest]
    hyperparams_of_interest = {key: hyperparams_of_interest[key] for key in hyperparams_of_interest.keys() if key not in empty_groups}
    if [] in groupings_of_interest:
        groupings_of_interest.remove([])
    print('groupings_of_interest =', groupings_of_interest)
    print('hyperparams_of_interest =', hyperparams_of_interest)
    print('  (removed empty groups: {0}.)'.format(', '.join(empty_groups)))

    # Output basic stats
    write_summary(dfs)

    print()

    if args.histories:
        # Create separate history loss & score plot for best logs
        print('Plotting top {0} training histories.'.format(num_best_to_plot_individually))
        best_model_names = dfs['all'].nlargest(num_best_to_plot_individually, score)['model_name']
        best_log_paths = ['/logs/'.join(os.path.split(name)) + '.log' for name in best_model_names]
        best_logs = read_logs(best_log_paths)
        for i, log in enumerate(best_logs):
            plot_loss_and_acc_history(log, out_path, score=score, index=i, show_plot=False)

    print()

    # Create boxplot of mean score per group
    print('Plotting boxplots of {0}.'.format(', '.join([str(grouping) for grouping in groupings_of_interest])))
    for grouping in groupings_of_interest:
        # Equal number of models for each group of interest (for fair comparison)
        print(grouping, [len(dfs[key]) for key in grouping])
        dfs_equalized_by_group = {}
        min_sample_size = min([len(dfs[key]) for key in grouping])
        for key in grouping:
            if len(dfs[key]) > min_sample_size:
                print('  Subsampling {0} from {1} runs in group {2}'.format(min_sample_size, len(dfs[key]), key))
                dfs_equalized_by_group[key] = dfs[key].sample(min_sample_size)
            else:
                dfs_equalized_by_group[key] = dfs[key]
        boxplot_means(dfs if args.full_samples else dfs_equalized_by_group, grouping, out_path, best_per_value=30, score=score, axes_ylim=None, show_plot=args.showplots)

    print()

    if args.histories:

        print('Plotting history plots of {0} given specified hyperparams.'.format(', '.join(hyperparams_of_interest.keys())))

        # For each group, each parameter of interest for that group, draw all histories in single plot:
        for group in hyperparams_of_interest.keys():

                for param in hyperparams_of_interest[group]:

                    # See if param_name should be plotted as caterogical or continuous:
                    plot_categorical = False
                    unique_values = list(dfs[group][param].unique())
                    if isinstance(unique_values[0], str) or isinstance(unique_values[0], np.bool_):
                        plot_categorical = True
                    if None in unique_values: unique_values.remove(None)
                    if 'None' in unique_values: unique_values.remove('None')

                    if len(unique_values) > 1:

                        print('  Plotting', group, 'given', param, unique_values, '(categorical)' if plot_categorical else '(continuous)')

                        # If categorical, first downsample to equally many samples for each param value
                        if plot_categorical:
                            dfs_per_value = [dfs[group].loc[df[param] == value] for value in unique_values]
                        if plot_categorical and not args.full_samples:
                            min_size = min([len(d) for d in dfs_per_value])
                            dfs_equalized_by_value = [d.sample(min_size) for d in dfs_per_value]
                            dfs_to_plot = dfs_equalized_by_value
                        else:
                            dfs_to_plot = [dfs[group]]

                        # Then optionally take only top N best runs:
                        if plot_categorical and num_best_to_plot_jointly_per_value:
                            dfs_to_plot = [df.nlargest(num_best_to_plot_jointly_per_value, score) for df in dfs_to_plot]

                        model_names_per_value = [df['model_name'] for df in dfs_to_plot]

                        # Plotting function wants a plain list of logs:
                        model_names_to_plot = [model_name for model_names in model_names_per_value for model_name in model_names]
                        log_paths_to_plot = ['/logs/'.join(os.path.split(name)) + '.log' for name in model_names_to_plot]
                        logs_to_plot = read_logs(log_paths_to_plot)

                        plot_history_per_hyperparam(logs_to_plot, param, out_path, score=score, plot_categorical=plot_categorical, show_plot=args.showplots, id=group)


# plt.hist([dfs['static']['val_macro_f1'], dfs['noentlib']['val_macro_f1']])
# plt.show()


def read_logs(log_path, recursive=True):

    if isinstance(log_path, str):
        if recursive:
            log_paths = glob.glob(os.path.join(log_path, '**', '*.log'), recursive=True)
        else:
            log_paths = glob.glob(os.path.join(log_path, '*.log'), recursive=False)
    else:
        log_paths = log_path    # i.e., it's already a list of paths.

    print('  Reading {0} log files...'.format(len(log_paths)))

    # Read all logs into dataframes, collect only those that are non-empty.
    logs = []
    for log_name in log_paths:
        log = _read_log(log_name)
        if len(log.index) != 0 and (len(log.folds) == log.params['folds']):
            logs.append(log)

    return logs


def write_logs_to_csv(logs, csv_path):

    print('Writing log file summaries to', csv_path + '.')

    # Store the settings and maxima from ALL logs in scores.csv file:
    with open(csv_path, "w", encoding="utf-8") as outf:
        param_names = list(logs[0].params.keys())
        score_names = list(logs[0].maxima.keys())
        param_names.sort()
        score_names.sort()
        print(','.join(param_names)+','+','.join(score_names) + ',model_name', file=outf)
        for log in logs:
            values = [str(log.params[param]) for param in param_names]
            maxima = [str(log.maxima[score]) for score in score_names]
            print(','.join(values) + ',' + ','.join(maxima) + ',' + log.log_name[:-4].replace('/logs/', '/'), file=outf)
    outf.close()


def read_csv(csv_path):

    print('Reading results from', csv_path + '.')

    df = pd.read_csv(csv_path)

    # Manual fix of the way dropout probs are written/read (problem with config utils, now fixed there)
    # df.loc[df['dropout_prob_1'] == 'False', 'dropout_prob_1'] = 0
    # df.loc[df['dropout_prob_2'] == 'False', 'dropout_prob_2'] = 0
    # df['dropout_prob_1'] = pd.to_numeric(df['dropout_prob_1'])
    # df['dropout_prob_2'] = pd.to_numeric(df['dropout_prob_2'])

    return df


def data_groups(df):
    # Construct sub-datasets:
    dfs = {}
    dfs['all'] = df

    dfs['noentlib'] = df.loc[df['entity_library'] == 'False']
    dfs['entlib'] = df.loc[df['entity_library'] != 'False']
    dfs['static'] = df.loc[df['entity_library'] == 'static']
    dfs['dynamic'] = df.loc[df['entity_library'] == 'dynamic']
    dfs['static_cos'] = dfs['static'].loc[df['gate_type'] == 'cos']
    dfs['static_dot'] = dfs['static'].loc[df['gate_type'] == 'dot']

    dfs['static_cos_noWS'] = dfs['static_cos'].loc[dfs['static_cos']['entlib_shared'] == 'False']
    dfs['static_dot_noWS'] = dfs['static_dot'].loc[dfs['static_dot']['entlib_shared'] == 'False']
    dfs['static_cos_WS'] = dfs['static_cos'].loc[dfs['static_cos']['entlib_shared'] == 'True']
    dfs['static_dot_WS'] = dfs['static_dot'].loc[dfs['static_dot']['entlib_shared'] == 'True']

    dfs['weights'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_weights'] == True]
    dfs['noweights'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_weights'] == False]

    dfs['valueweights'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_value_weights'] == 'True']
    dfs['valuezeros'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_value_weights'] == 'False']

    dfs['keys'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_key'] == 'True']
    dfs['nokeys'] = dfs['dynamic'].loc[dfs['dynamic']['entlib_key'] == 'False']

    dfs['weights_nokeys'] = dfs['weights'].loc[dfs['weights']['entlib_key'] == 'False']
    dfs['noweights_nokeys'] = dfs['noweights']

    dfs['dot'] = df.loc[df['gate_type'] == 'dot']
    dfs['cos'] = df.loc[df['gate_type'] == 'cos']
    dfs['mlp'] = df.loc[df['gate_type'] == 'mlp']
    dfs['softmax'] = df.loc[df['gate_softmax'] == True]
    dfs['nosoftmax'] = df.loc[df['gate_softmax'] == False]

    return dfs


def write_summary(dfs):
    df = dfs['all']
    # TODO make more insightful
    print(df.describe()['val_macro_f1'])
    #Best models:
    pd.options.display.max_colwidth = 300
    print(df.nlargest(10, 'val_macro_f1')[['model_name','val_macro_f1', 'val_macro_f1_main']])
    print('\n')


def check_normality(dfs, labels, scores, verbose=False):
    result = True
    for label in labels:
        for score in scores:
            value, p = stats.normaltest(dfs[label][score])
            result = result and p >= 0.05
            if verbose:
                print('{3}-{4}: {0}likely normal ({1},{2})'.format('un' if p < 0.05 else '', value, p, label, score))
    return result


def _split_folds(log):
    """
    Splits a pandas dataframe with appropriate metadata into a list of dataframes
    according to column 'fold'. Each new dataframe inherits the relevant metadata.
    :param log: pandas dataframe with at least columns 'fold' and 'val_acc' and appropriate metadata.
    :return: a list of pandas dataframes, one for each fold.
    """
    folds = []
    for i in log.fold.unique():
        fold = log[log.fold==i]
        fold.params = log.params
        fold.log_name = log.log_name
        fold.fold_num = i
        fold['val_loss'] = pd.to_numeric(fold['val_loss'], errors = 'coerce')
        # saved_epoch = fold['val_loss'].idxmin()
        fold.best_epoch = fold['val_macro_f1'].idxmax()
        if str(fold.best_epoch) != 'nan':
            fold.fold_maxima = {label: float(fold[label][fold.best_epoch]) for label in maxima_of_interest}
            folds.append(fold)
    return folds


def _read_log(log_path):
    """
    Reads a whitespace-separated, #-commented log file with at least columns 'val_acc' and 'fold' and returns a pandas dataframe.
    :param log_path: path to appropriate log file
    :return: pandas dataframe of the log with various metadata attributes.
    """
    # Read the first line and the data
    log = pd.read_csv(log_path,
                      #delim_whitespace=True,
                      sep="\t",
                      comment='#',
                      index_col=0)
    settings, _, _ = config_utils.settings_from_config(log_path.replace('.log', '.ini'), random_sample=False)

    # Add various metadata, and split the log into the different folds.
    with warnings.catch_warnings():
        # Ignore warnings; Pandas is afraid we're trying to create new columns.
        warnings.simplefilter("ignore")
        log.log_name = log_path
        log.params = vars(settings.model)
        log.params.update(vars(settings.training))
        log.params.update(vars(settings.data))

        # Add folds as convenient metadata -- although this (presumably) involves duplicating the data...
        log.folds = _split_folds(log)
        log.maxima = {}
        log.variances = {}
        for label in maxima_of_interest:
            best_scores_in_log = [fold.fold_maxima[label] for fold in log.folds]
            log.maxima[label] = np.mean(best_scores_in_log)
            log.variances[label] = np.var(best_scores_in_log)
        log.best_epochs = [fold.best_epoch for fold in log.folds]
        log.best_epoch_mean = np.mean(log.best_epochs)
        log.best_epoch_var = np.var(log.best_epochs)
    return log



def boxplot_means(dfs, labels, out_path, best_per_value=None, score='val_macro_f1', axes_ylim=None, show_plot=False):

    if best_per_value is not None:
        dfs = {key: dfs[key].nlargest(best_per_value, score) for key in labels}

    plottitle = '{0}\n{1} models of each type'.format(score,
                                                           'best '+str(best_per_value) if best_per_value is not None else 'random')
    plotname = 'plot_{0}_{1}_{2}.png'.format(score,
                                            'top'+str(best_per_value) if best_per_value is not None else 'rnd',
                                            '-'.join(labels))

    if axes_ylim is None:
        axes_ylim = [0, 100]

    plt.style.use('ggplot')
    plt.title(plottitle)
    axes = plt.gca()
    axes.set_ylim(axes_ylim)
    axes.set_ylabel(score)

    plt.boxplot([dfs[label][score] for label in labels],
                labels=labels)

    plotpath = os.path.join(out_path, plotname)
    plt.savefig(plotpath)
    print('  Boxplot saved to {0}'.format(plotpath))

    compare_means(dfs, labels, score)
    if show_plot:
        plt.show()

    plt.close()


def compare_means(dfs, labels, score):
    normal = check_normality(dfs, labels, [score])
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i < j:
                if normal:
                    test = stats.ttest_ind(dfs[label1][score], dfs[label2][score], equal_var=False)
                else:
                    test = stats.ks_2samp(dfs[label1][score], dfs[label2][score])
                print('    {5} {0} --> {1}: {2:.2f} (p-value {3:.4f}; {4})'.format(label1,
                                                                            label2,
                                                                            test.statistic,
                                                                            test.pvalue,
                                                                            'unequal variance T-test' if normal else 'Kolmogorov-Smirnov test',
                                                                            '***' if test.pvalue < .001 else
                                                                               ' **' if test.pvalue < .01 else
                                                                               '  *' if test.pvalue < .05 else
                                                                               '   ',
                                                                            )
                      )

def plot_loss_and_acc_history(log, out_path, score='val_macro_f1', index=None, show_plot=False):
    """
    Creates a plot from the log, of the columns 'train_loss', 'val_loss', 'train_acc',
    'val_acc', through the epochs, with a separate line for each fold.
    :param log: a pandas dataframe with appropriate columns and metadata from read_log().
    :return: a matplotlib.pyplot object of the training history
    """
    score_suffix = '_'.join(score.split('_')[1:])      # e.g., take 'macro_f1' from 'val_macro_f1'.

    plot_losses = ['train_loss', 'val_loss']
    plot_accuracies = ['train_' + score_suffix, 'val_' + score_suffix]
    color_maps = [cm.Blues, cm.Reds]  # blue-ish colors for train loss/acc; red-ish colors for val_loss/acc

    # Setup of the plots
    fig, ax1 = plt.subplots(figsize=(10,8))
    plt.title('Training history'+('' if index is None else ' of {0}th best log'.format(index)))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()   # This overlays two plots with same x axis but different y axes
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0,100)
    normalize = mcolors.Normalize(vmin=0, vmax=100) # A color normalizer to be used below

    # Plot the data in appropriate colors
    first = True    # Lazy bookkeeping: only add the first plots to the figure's legend.
    for fold in log.folds:
        maximum = fold.fold_maxima[plot_accuracies[1]]
        for loss, colormap in zip(plot_losses, color_maps):
            ax1.plot(fold[[loss]], '-', label=loss if first else '_nolegend_', color=colormap(normalize(maximum)))
        for acc, colormap in zip(plot_accuracies, color_maps):
            ax2.plot(fold[[acc]], '--', label=acc if first else '_nolegend_', color=colormap(normalize(maximum)))
        first = False

    # Create a single legend for both overlaid plots
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc=0)

    plot_path = os.path.join(out_path, 'history_{0}__{1}{2}.png'.format(score,
                                                                        (str(index) + '_') if index is not None else '',
                                                                        os.path.basename(log.log_name)[:-4]))
    if show_plot:
        plt.show()
    plt.savefig(plot_path)
    print('  History plot saved to {0}.'.format(plot_path))
    plt.close()


def plot_history_per_hyperparam(logs, param_name, out_path, score='val_macro_f1', plot_categorical=False, show_plot=False, id=None):
    """
    Creates a plot from a set of logs, of 'val_acc' through the epochs, for each fold for each model.
    Plot contains a separate line for each fold, colored according to hyperparameter value, and with
    line transparency encoding the variance ('unreliability') across folds for a given model.
    :param logs: a list of pandas dataframes with appropriate columns and metadata, from read_log().
    :param param_name: a hyperparameter name, such as 'hidden_size'.
    :return: a matplotlib.pyplot object of the training histories colored by hyperparameter name.
    """

    random.shuffle(logs)

    # Create three lists/arrays of the same size, namely the total number of runs (models * folds)
    accuracies = [fold[[score]] for log in logs for fold in log.folds]  # to be plotted
    variances = np.array([log.variances[score] for log in logs for _ in log.folds])  # determines line transparency
    param_values = [log.params[param_name] for log in logs for _ in log.folds]  # determines line color

    opacities = (1.0 - (variances / variances.max())) ** 2  # transparency to indicate variance in max accuracy

    # Setup the plot
    plt.figure(figsize=(12, 8))
    # Further layout
    plt.xlabel('Epoch')
    plt.ylabel(score)
    plt.title("{0} history depending on the '{1}' hyperparameter.".format(score, param_name))
    plt.gca().set_ylim(0, 100)

    if plot_categorical:

        colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "black"]
        unique_values = list(set(param_values))
        unique_values = [str(v) for v in unique_values]
        unique_values.sort()
        unique_values = [((v == 'True') if (v == 'True' or v == 'False') else v) for v in unique_values]
        colors_mapping = {x: colors[i] for i, x in enumerate(unique_values)}

        already_used_labels = set()
        for accuracy, val, opacity in zip(accuracies, param_values, opacities):
            if val in already_used_labels:
                plt.plot(accuracy, linewidth=.5, alpha=opacity*0.8, color=colors_mapping[val])
            else:
                plt.plot(accuracy, linewidth=.5, alpha=opacity*0.8, color=colors_mapping[val], label=val)
                already_used_labels.update({val})

        plt.legend()

    else:
        colormap = cm.viridis    # colors to indicate the hyperparameter value
        if param_name in plot_logarithmic:
            normalize = mcolors.LogNorm(vmin=min(param_values), vmax=max(param_values))
        else:
            normalize = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))

        # plot every accuracy history with the appropriate color and opacity.
        for accuracy, param_value, opacity in zip(accuracies, param_values, opacities):
            plt.plot(accuracy, color=colormap(normalize(param_value)), alpha=opacity*0.8, linewidth=.5, label=str(param_value))

        # setup the colorbar
        scalarmappable = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappable.set_array([])    # Nonsense but necessary.
        cb = plt.colorbar(scalarmappable)
        cb.ax.set_title(param_name)

    plot_path = os.path.join(out_path, 'history_{0}__{2}given_{1}.png'.format(score, param_name, id + '_' if id is not None else ''))
    plt.savefig(plot_path)
    print('  History plot saved to {0}.'.format(plot_path))
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    _main()

