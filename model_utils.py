import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import data_utils
import distance

import warnings
from sklearn.metrics import accuracy_score, f1_score


# TODO @Future: include in logs precision and recall.
def train(model, inputs, targets, indices, val_inputs, val_targets, idx_to_word, word_to_idx, word_vectors, settings, no_shuffle, logger):
    """
    Trains a model on training data X, y, testing on validation data,
    with various settings given by args. If no validation set is given,
    evaluates only on training data.
    :param model:
    :param X: Numpy array containing inputs (in original order)
    :param y: Numpy array containing targets (in original order, with NON-ENTITY_MENTION for non-mentions)
    :param train_sequence_bounds: List of pairs of [start, end] delineating training scenes/episodes in X/y.
    :param validation_sequence_bounds: List of pairs of [start, end] delineating validation scenes/episodes in X/y.
    :param settings: forwarded from the argument parser in main.py.
    :return:
    """

    # Store arguments in convenient variables
    shuffle_data = not no_shuffle
    use_validation_data = len(val_inputs)
    use_cuda = next(model.parameters()).is_cuda

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    params = filter(lambda p: p.requires_grad, model.parameters())
    # TODO @Future: improve if more (e.g., AdaGrad) optimizers want to be tried
    # TODO @Future: Apparently some have argued that weight decay shouldn't apply to embeddings... maybe try this?
    if settings.optimizer == 'adadelta':
        optimizer = optim.Adadelta(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    else:
        optimizer = optim.Adam(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    #elif args.useSGD: optimizer = optim.SGD(params, lr=args.lr) # TODO specify momentum?

    # Variables for bookkeeping during training
    epoch = -1  # Start at -1 in order to compute pre-training scores.
    iteration = 0
    best_com_acc = 0
    best_model = model
    num_epochs_no_improve = 0

    while epoch < settings.epochs:

        logger.say('Epoch: '+ str(epoch))

        if epoch != -1:

            model.train()

            # Iterate through batches (lists of batches of chunks), don't care about the order (hence the '_')
            for batch_ids in data_utils.batches(indices, settings.batch_size, shuffle=shuffle_data):

                # Wrap inputs in a variable; compute mask of where outputs are desired
                batch_inputs_var = autograd.Variable(inputs[batch_ids], requires_grad=True)
                batch_targets_var = autograd.Variable(targets[batch_ids], requires_grad=False)
                batch_seed_var = autograd.Variable(torch.from_numpy(batch_ids).cuda(), requires_grad=True)

                # Apply model and calculate loss
                hidden = model.init_hidden(len(batch_ids))
                if use_cuda:
                    hidden = hidden.cuda()
                model.zero_grad()
                loss = 0

                max_len = batch_inputs_var.size()[1]

                for c in range(max_len):
                    output, hidden = model(batch_inputs_var[:, c], hidden, seed=batch_seed_var if c == 0 else None)
                    loss += loss_function(output.view(len(batch_ids), -1), batch_targets_var[:, c])

                loss.backward(retain_graph=False)
                optimizer.step()

                iteration += 1

        # Every N epochs, collect performance loggerstatistics:
        if (epoch + 1) % settings.test_every == 0:

            test_indices = indices      # TODO restrict

            model.eval()

            # Reserve a dictionary to store the various results in
            performance = {"epoch": epoch, "iteration": iteration,}
                           # "training": None,
                           # "validation": None}

            # Obtain predictions and loss on training data; predictions are in order.
            train_predictions = np.array(get_predictions(model, test_indices, batch_size=settings.batch_size, shuffle=False))

            # Get all scores and insert them into the dictionary
            performance.update(get_scores(train_predictions, test_indices, idx_to_word, word_to_idx, word_vectors))

            logger.log(performance)

            # Also keep track of the relative performance increase/decrease:
            # TODO keep track of difference
            # f1_diff_train = performance["training"]["macro_f1_score"] - prev_train_score
            # prev_train_score = performance["training"]["macro_f1_score"]

            # And do the same for validation data:
            # TODO validation
            # if use_validation_data:
            #     val_predictions, val_mean_loss = get_predictions(model, val_inputs, val_targets, batch_size=settings.batch_size)
            #
            #     val_scores = get_scores(val_predictions, val_targets)
            #     performance["validation"].update(val_scores)
            #     performance["validation"]["loss"] = val_mean_loss
            #     f1_diff_val = performance[stop_criterion_data]["macro_f1_score"] - prev_val_score
            #     prev_val_score = performance[stop_criterion_data]["macro_f1_score"]

            # Print the various scores
            # logger.say(performance)
            # Keep track of best performance, and assess whether to stop training
            if performance['avg_com_acc'] > best_com_acc:    # i.e., if the model is improving.
                num_epochs_no_improve = 0
                best_model = model
                best_com_acc = performance['avg_com_acc']
            else:                               # i.e., if the model is NOT improving.
                num_epochs_no_improve += settings.test_every

            if num_epochs_no_improve >= settings.stop_criterion:
                message = 'Stopped after epoch {0} because com acc did not improve for {1} epochs.'.format(epoch, num_epochs_no_improve)
                logger.say(message)
                logger.log('# ' + message)
                break

        epoch += 1

    return model, best_model


def get_predictions(model, indices, batch_size=100, shuffle=True):

    model.eval()
    predictions = []

    for batch_indices in data_utils.batches(indices, batch_size, shuffle=shuffle):

        with torch.no_grad():

            batch_outputs = model.generate(batch_indices, predict_len=20)
            preds = [pred[1:].split('<')[0] for pred in batch_outputs]

        predictions.extend(preds)

    return predictions


def get_scores(predictions, indices, idx_to_word, word_to_idx, word_vectors):
    """
    Computes prediction accuracy, F1 scores for all/main entities, macro average of F1 scores for all/main entities.
    :param predictions: list of predicted labels
    :param targets: list of true labels
    :param restrict_indices: numpy array containing indices (if any) to restrict the scores to (e.g., only pronouns).
    :return: dictionary of all scores
    """
    # TODO @Future: More of this could be done on gpu (e.g.: https://www.kaggle.com/igormq/f-beta-score-for-pytorch/code )

    # for i, ind in enumerate(indices[100:200]):
    #     print(idx_to_word[ind].ljust(18), predictions[i+100].ljust(18))

    str_sims = []
    sem_sims = []
    com_accs = []

    for i, pred in zip(indices, predictions):
        str_sim = 1.0 - (distance.levenshtein(idx_to_word[i], pred) / max(len(idx_to_word[i]), len(pred)))

        if str_sim == 1:
            sem_sim = 1
        else:
            j = word_to_idx.get(pred)
            if j is None:
                sem_sim = 0
            else:
                sem_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(word_vectors[i]).cuda(), torch.from_numpy(word_vectors[j]).cuda(), dim=0).item()

        com_acc = max(str_sim, sem_sim)  # TODO Make smarter, e.g., semantic closeness of closest string(s).

        str_sims.append(str_sim)
        sem_sims.append(sem_sim)
        com_accs.append(com_acc)

        # if sem_sim < 1.0 and sem_sim > 0.0:
        #     print('{0} {1} ({2:.2f}, {3:.2f})'.format(idx_to_word[i], pred, str_sim, sem_sim))

    avg_str_sim = sum(str_sims) / len(str_sims)
    avg_sem_sim = sum(sem_sims) / len(sem_sims)
    avg_com_acc = sum(com_accs) / len(com_accs)

    # avg_inter_str_dist, avg_inter_sem_dist, comp = correlation(decoder.seed.weight, predictions,
    #                                                            preview_inds)  # TODO not feasible to do for all words; reuse the semantic one.

    all_scores = {'acc': (idx_to_word[indices] == predictions).mean(),
                  'avg_str_sim': avg_str_sim,
                  'avg_sem_sim': avg_sem_sim,
                  'avg_com_acc': avg_com_acc,
                  'total': len(indices)}

    return all_scores
