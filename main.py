import copy
import os.path
import output_utils
import sys
import torch
import numpy as np
import argparse
import time
import data_utils
import models
import model_utils
import config_utils
import sklearn.metrics
import pandas as pd # Try to avoid?
from matplotlib import pyplot as plt
from collections import Counter, defaultdict

# torch.cuda.empty_cache()

def main():

    args, settings = parse_args_and_settings()
    logger = output_utils.Logger(args)
    logger.shout('python main.py ' + ' '.join(sys.argv[1:]))

    num_threads = 5
    torch.set_num_threads(num_threads)

    if args.phase == 'train' or args.phase == 'deploy':

        # Loading train data is needed for train, but also for deploy, namely if vocabulary doesn't exist yet:
        # for i in range(0,100,10):
        #     print('computing compositionality of generation',i)
        #     settings.data.dataset = 'models/2018_07/data/fixed--2018_07_09_18_41_38__{0}.txt'.format(i)
        idx_to_word, word_to_idx, word_vectors = data_utils.get_data(settings.data, args.stats)
        idx_to_word_original = idx_to_word

    if args.phase == 'train':
        logger.save_config(settings.orig)
        logger.say(output_utils.bcolors.BOLD + 'Training on ' + settings.data.dataset)

        reproduction_command = 'python main.py ' + '-c ' + os.path.join(logger.log_dir, logger.run_name + '.ini')
        logger.shout(reproduction_command)
        logger.log('# ' + reproduction_command)
        logger.log(
            'epoch\titeration\tfold\ttrain_loss\ttrain_acc\ttrain_macro_f1\ttrain_macro_f1_main\ttrain_total\tval_loss\tval_acc\tval_macro_f1\tval_macro_f1_main\tval_total\tmodel')

        if settings.data.generations > 1:
            logger.generation = 'START'
            logger.save_word_vectors(idx_to_word, word_vectors)  # Save duplicate of the data on which the first model will be trained

        for i in range(settings.data.generations):

            print("Generation", i)

            logger.generation = i

            bottleneck = round(settings.training.bottleneck * len(idx_to_word))
            training_ids = np.random.choice(np.arange(len(idx_to_word)), bottleneck, replace=False, p=None)

            best_model = run_training(settings, idx_to_word, word_to_idx, word_vectors, training_ids, logger, not args.no_cuda)

            predictions = model_utils.get_predictions(best_model, np.arange(len(idx_to_word)),
                                                      batch_size=settings.training.batch_size, shuffle=False)

            # for next generation:
            idx_to_word = np.array(predictions)
            word_to_idx = {idx_to_word[i]: i for i in range(len(idx_to_word))}
            logger.save_word_vectors(idx_to_word, word_vectors)     # Save the data on which the next model will be trained

            for i in range(0,300,5):
                print('* ' if i not in training_ids else '  ', idx_to_word_original[i].ljust(18), idx_to_word[i].ljust(18))

            # TODO Add computation of ground truth scores as well.

    if args.phase == 'deploy':
        logger.say(output_utils.bcolors.BOLD + 'Deploying ' + str(len(args.model)) + ' models (' + (
            args.run_name if len(args.model) > 1 else args.model[0]) + ')...\n   ...on ' + ('folds of ' if not args.no_cv else '') + args.deploy_data)
        args.answer_file, with_keys = run_deploy(args.model, settings, args.deploy_data, args.answers_per_fold, args.no_cv, logger, not args.no_cuda)
        # After deploying, evaluate (unless not desired or data does not contain reference keys):
        if not args.no_eval:
            if with_keys is True:
                args.phase = 'evaluate'
            else:
                logger.shout('Warning: Model predictions will not be evaluated, since given data does not contain reference labels. ')

    if args.phase == 'evaluate':
        logger.say(output_utils.bcolors.BOLD + 'Evaluating ' + ('(not SemEval style) ' if args.no_semeval else '(SemEval style) ') + 'predictions of ' + args.answer_file)
        run_evaluate(args.answer_file, args.deploy_data, logger)


def run_training(settings, idx_to_word, word_to_idx, word_vectors, training_ids, logger, use_cuda):

    inputs, targets = data_utils.to_char_tensors(idx_to_word)

    # fold_indices = range(settings.data.folds)
    # if settings.data.folds > 1:
    #     pass  # TODO divide into folds
    # else:
    #     # No cross-validation:
    #     val_inputs = torch.from_numpy(np.array([]))
    #     val_targets = torch.from_numpy(np.array([]))
    #
    # for fold_idx in fold_indices:
    #     # For bookkeeping (logging five folds in one file):
    #     logger.fold_idx = fold_idx
    #
    #     # Select training and (if cross-validation) validation data:
    #     if settings.data.folds > 1:
    #         training_domains = folds[:fold_idx]+folds[fold_idx+1:]
    #         inputs = torch.from_numpy(np.concatenate(tuple([domain['X'] for domain in training_domains]))).float()
    #         targets = torch.from_numpy(np.concatenate(tuple([domain['y'] for domain in training_domains])))
    #         val_inputs = torch.from_numpy(folds[fold_idx]['X']).float()
    #         val_targets = torch.from_numpy(folds[fold_idx]['y'])

    # Initialise model
    model = models.CharRNN(settings.model, data_utils.n_characters, torch.Tensor(word_vectors), logger=logger)

    # CUDAfy model and datasets:
    if use_cuda:
        model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
        # val_inputs = val_inputs.cuda()
        # val_targets = val_targets.cuda()

    # Train the model
    last_model, best_model = model_utils.train(model, inputs, targets, training_ids, [], [], idx_to_word, word_to_idx, word_vectors, settings.training,
                                               settings.training.no_shuffle, logger=logger)

    # Save the best model through the logger
    logger.save_model(best_model)

    return best_model


def run_deploy(model_path, settings, data_path, answers_per_fold, no_cv, logger, use_cuda):

    data = data_utils.get_data(settings.data)

    # Load all models from model_path:
    model_list = []
    for path in model_path:
        model_fold = models.Clusterer(settings.model, logger=logger)
        model_fold.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        if use_cuda:
            model_fold.cuda()
        model_list.append(model_fold)
        logger.whisper('Loaded model from ' + path)

    print(model_list[0])

    if no_cv:   # To deploy all models as ensemble to all the data

        input_vecs = torch.from_numpy(np.concatenate(tuple([domain['X'] for domain in data])))
        targets = torch.from_numpy(np.concatenate(tuple([domain['y'] for domain in data])))

        collect_ensembles_preds = False     # TODO @Future This may be useful at some point.

        predictions_zipped, _ = model_utils.get_indexed_predictions_with_targets(
            model_list, input_vecs, targets, use_cuda,
            collect_ensembles_preds=collect_ensembles_preds)

        # Write answers through logger
        answers_path = logger.write_answers_csv(data_path, predictions_zipped, model_suffix="--ensemble", config=settings.orig)

        # Optionally also per individual model (i.e., each model trained on one fold):
        if answers_per_fold:
            for i, model in enumerate(model_list):
                predictions_zipped, _ = model_utils.get_indexed_predictions_with_targets(
                    model, input_vecs, targets, use_cuda)
                logger.write_answers_csv(data_path, predictions_zipped, model_suffix='--fold'+str(i))

    else:   # To deploy per fold of the data
        results = []
        keys = []
        folds = data
        for fold_idx in range(settings.data.folds):

            val_inputs = torch.from_numpy(folds[fold_idx]['X']).float()
            val_targets = torch.from_numpy(folds[fold_idx]['y'])
            # CUDAfy:
            if use_cuda:
                val_inputs = val_inputs.cuda()
                val_targets = val_targets.cuda()

            # Obtain predictions for this fold:
            predictions = model_utils.get_predictions_and_mean_loss(model_list[fold_idx], val_inputs, val_targets, batch_size=settings.training.batch_size, shuffle=False)

            if use_cuda:
                predictions = predictions.cpu()
                val_targets = val_targets.cpu()

            predictions = predictions.numpy()
            val_targets = val_targets.numpy()

            # Optionally write answers for this one fold
            if answers_per_fold:
                logger.write_answers_csv(settings.data.dataset + "--fold" + str(fold_idx), [predictions], [val_targets],
                                         model_suffix="--fold" + str(fold_idx), config=settings.orig)

            # But also store them, to be merged and sorted later, for writing merged answers

            results.append(predictions)
            keys.append(val_targets)

        # Write answers merged over all folds through logger
        answers_path = logger.write_answers_csv(settings.data.dataset, results, keys, model_suffix="--cv", config=settings.orig)

    return answers_path, True


def run_evaluate(answer_file, deploy_data, logger):

    df = data_utils.read_answers_csv(answer_file, logger=logger)

    # TODO: precision and recall scores per seq index; scores per domain; scores per cluster index.

    predictions = df.loc['predictions']
    targets = df.loc['targets']

    correct = predictions == targets

    mean_acc_per_domain_per_seq_idx = correct.groupby(level=['dom']).mean()
    mean_acc_per_seq_idx = correct.mean()
    mean_acc = mean_acc_per_seq_idx.mean()
    mean_acc_per_domain = mean_acc_per_domain_per_seq_idx.mean(axis=1)

    # All first points:
    score = sklearn.metrics.f1_score

    scores_per_pos = np.stack([score(targets.loc[:,i].values.flatten(), predictions.loc[:,i].values.flatten(), labels=[0,1,2], average=None) for i in range(0,30)])

    freq_per_pos = np.stack([(predictions == 0).mean(), (predictions == 1).mean(), (predictions == 2).mean()], axis=1)

    from cycler import cycler

    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    plt.plot(scores_per_pos[:, 0], '-', label='score 0')
    plt.plot(scores_per_pos[:, 1], '-', label='score 1')
    plt.plot(scores_per_pos[:, 2], '-', label='score 2')
    plt.plot(freq_per_pos[:, 0], '--', label='freq 0')
    plt.plot(freq_per_pos[:, 1], '--', label='freq 1')
    plt.plot(freq_per_pos[:, 2], '--', label='freq 2')

    plt.legend()
    plt.show()

    # # TODO @Future: Offload file naming, structure and writing etc. to logger.
    # output_file_scores = answer_file.strip('.csv') + '_scores.txt'
    # output_file_matrix = answer_file.strip('.csv') + '_matrix.csv'
    #
    # with open(output_file_scores, 'w') as file:
    #     for score_type in sorted(scores):
    #         file.write(score_type.upper() + '\n')
    #         for measure in sorted(scores[score_type]):
    #             file.write(measure + ':\t' + str(scores[score_type][measure]) + '\n')
    #
    #         file.write('\n\n')
    #
    #     file.write(data_utils.data_summary(evaluate_data))
    #
    # with open(output_file_matrix, 'w') as file:
    #     confusion_mat.to_csv(file, sep='\t')
    #
    # logger.whisper('Scores written to ' + output_file_scores)
    # logger.whisper('Confusion matrix written to ' + output_file_matrix)


def parse_args_and_settings():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--phase", default=None,
                        help="can be train, deploy (by default includes evaluate) or evaluate. Automatically inferred if not specified.")

    # Only for training:
    # TODO @Future Make parser give warnings when wrong combination of arguments is given (standard feature of argparse perhaps?)
    parser.add_argument("-c", "--conf_file", default=None,
                        help="Path to config file including .ini; can be left None only in deploy/evaluation, in which case it is derived from model/answers path.",
                        metavar="FILE")
    parser.add_argument("-r", "--random", action='store_true',   # Note: also kinda used, but overwritten, during deploy/evaluate
                        help="To sample values randomly from intervals in config file.")
    parser.add_argument("-d", "--subdir", default=None,   # Note: also used, but overwritten, during deploy
                        help="(training phase) use /models/<subdir> for output; by default set to year_month.")
    parser.add_argument("-n", "--run_name", default=None,  # Note: also used, but overwritten, during deploy/evaluate
                        help="(training phase) by default set to time stamp; always automatically appended by randomly sampled params.")

    # For deployment/evaluation:
    parser.add_argument("-t", "--deploy_data", default=None,
                        help="Default is data on which the model was trained; can be 'train', 'test' or 'trial', abbreviations as defined in config_utils. During evaluation this can be omitted: it will be read from the predictions .csv file.")
    parser.add_argument("-l", "--deploy_level", default=None,
                        help="Scene or episode; default is the level on which the model was trained. During evaluation this can be omitted: it will be read from the predictions .csv file.")

    # Only for deployment
    parser.add_argument("-m", "--model", default=None,
                        help="(deployment phase) path to model, with base name (though without .pt suffix); if fold number is included, only that fold is considered.")
    parser.add_argument("--answers_per_fold", action='store_true',
                        help="To write an answers.txt file for each fold separately (in addition to their merger).")
    parser.add_argument("--no_cv", action='store_true',
                        help="Option to prevent respecting cross-validation (which is done by default when deployed on training data).")
    parser.add_argument("-s", "--no_eval", action='store_true',
                        help="If data includes keys, evaluation phase is run automatically after deployment; this option prevents that.")

    # Only for evaluation
    parser.add_argument("-a", "--answer_file", default=None,
                        help="Evaluates an answer file, outputting interesting statistics.")
    parser.add_argument("--no_semeval", action='store_true',
                        help="To turn off the SemEval filter for evaluation (filter which groups infrequent (< 3) entities together as 'other').")

    # Meta:
    parser.add_argument("--no_cuda", action='store_true',
                        help="Forces not using cuda; otherwise cuda is used whenever available.")
    parser.add_argument("-v", "--verbosity", type=int, default=3,
                        help="Sets verbosity regarding console output (default 3; lower to print less).")
    parser.add_argument("-f", "--no_files", action='store_true',
                        help="Prevents generation of folders and files (log, model, answer).")
    parser.add_argument("--stats", action='store_true',
                        help="Whether to printout k-means 'gold standard' for some of the clusterings.")
    parser.add_argument('-i', "--iterate", action='store_true')

    args = parser.parse_args()

    # If phase is not specified, this can usually be inferred from other arguments:
    if args.phase is None:
        if args.model is not None:
            args.phase = 'deploy'
        elif args.answer_file is not None:
            args.phase = 'evaluate'
        else:
            args.phase = 'train'

    # Use CUDA only if available:
    if not args.no_cuda and not torch.cuda.is_available:
        print('WARNING: CUDA requested but unavailable; running on cpu instead.')
        args.no_cuda = True

    # Deploy either a single model or a set of models (of the same type).
    # Also, from the model file arg.model also extract model_dir and run_name:
    if args.phase == 'deploy':
        if '.pt' in args.model:
            # A single model file .pt was provided, so deploy only on that:
            runs_path, args.run_name = os.path.split(args.model)
            args.model_dir, args.subdir = os.path.split(runs_path)
            args.run_name = args.run_name[:-3] # removes the .pt
            if '--fold' in args.run_name:
                args.run_name = args.run_name.split('--fold')[0]
            args.model = [args.model]
        else:
            # model name doesn't contain .pt (i.e., either directory, or directory+run_name:
            if os.path.isdir(args.model):
                # model is a directory
                runs_path = args.model
                args.run_name = None    # To be extracted below
            else:
                # model is not a directory, nor .pt; hence only run_name of model is given:
                runs_path, args.run_name = os.path.split(args.model)
            args.model_dir, args.subdir = os.path.split(runs_path)
            # Get all model paths from directory (with run_name):
            models = []
            for file in os.listdir(runs_path):
                if file.endswith(".pt"):
                    if args.run_name is None:
                        args.run_name = file[:-3]  # removes the .pt
                        if '--fold' in args.run_name:
                            args.run_name = args.run_name.split('--fold')[0]
                    if file.startswith(args.run_name):
                        models.append(os.path.join(runs_path, file))
                    elif os.path.isdir(args.model):
                        print("ERROR: run_name could not be inferred; directory contains multiple runs.\n Rerun with more specific --model (i.e., including model file name, minus .pt and minus --fold#).")
                        quit()
            args.model = sorted(models)

    # When evaluating, obtain run name etcetera from the provided answers .csv file:
    if args.phase == 'evaluate':
        args.run_name = os.path.basename(args.answer_file)[:-4]
        if args.run_name.endswith('--ensemble'):
            args.run_name = args.run_name[:-10]  # removes the --ensemble suffix
        if '--fold' in args.run_name:
            args.run_name = args.run_name.split('--fold')[0]
        if '--cv' in args.run_name:
            args.run_name = args.run_name.split('--cv')[0]
        args.model_dir = None  # This is kinda ugly.

    # For train phase a config file is mandatory; otherwise it can be automatically obtained:
    if args.conf_file is None:
        if args.phase == 'train':
            print('ERROR: training requires a config file (try including -c config.ini)')
            quit()
        elif args.phase == 'deploy':
            args.conf_file = os.path.join(runs_path, args.run_name + '.ini')
        elif args.phase == 'evaluate':
            args.conf_file = os.path.join(os.path.dirname(args.answer_file), args.run_name + '.ini')

    # Read the config file (either given as argument, or obtained from pre-trained model or its predictions file:
    if args.phase == 'deploy' or args.phase == 'evaluate':
        # Of course don't randomly sample when deploying or evaluating.
        args.random = False

    settings, fixed_params, sampled_params = config_utils.settings_from_config(args.conf_file, args.random)
    # NOTE: Which params were fixed or sampled determines the subdir and run_name in case of training.

    # If no level and data for deployment are given, these are taken from training data/level in config file
    if args.phase == 'deploy':
        args.deploy_data = args.deploy_data or settings.data.dataset

    # For evaluate, if deploy_data is not provided, attempt to read it from the answer_file:
    # (The alternative, of reading it from directory structure, seems too unsafe.)
    if args.phase == 'evaluate' and args.deploy_data is None:
        with open(args.answer_file) as file:
            firstline = file.readline()
            if firstline.startswith('#'):
                args.deploy_data = firstline.strip('# \n')

    # When deploying on a new dataset (not training data), cross-validation doesn't apply:
    if args.deploy_data != settings.data.dataset:
        args.no_cv = True

    # When training, create runs dir, id and run name if none were given (mostly time stamps).
    if args.phase == 'train':
        args.model_dir = 'models'       # Default for training output.
        args.subdir = args.subdir or time.strftime("%Y_%m")
        args.run_name = args.run_name or time.strftime("%Y_%m_%d_%H_%M_%S")
        if not sampled_params:
            args.run_name = 'fixed--' + args.run_name
        else:
            # TODO @Future: Automatic run naming can be considerably improved wrt. readability.
            sampled_params_strings = sorted([k[0:3] + "--" + str(sampled_params[k])[0:5].replace(",", "-") for k in sampled_params])
            args.run_name = '{0}--{1}'.format(args.run_name, "--".join(sampled_params_strings))

    # Within the settings Namespace, which is subject to overwriting, make sure to include a backup,
    # so that original settings can at any time be saved to a new config file.
    settings.orig = copy.deepcopy(settings)

    return args, settings


if __name__ == "__main__":

    main()