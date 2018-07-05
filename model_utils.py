import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import data_utils
import warnings
from sklearn.metrics import accuracy_score, f1_score


# TODO @Future: include in logs precision and recall.
def train(model, inputs, targets, indices, val_inputs, val_targets, settings, no_shuffle, logger):
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
    min_val_loss = 1000.0
    max_val_macro_f1 = -1.0
    max_train_macro_f1 = -1.0
    best_model = model
    prev_val_score = 0.0
    prev_train_score = 0.0
    num_epochs_no_improve = 0
    stop_criterion_data = "validation" if use_validation_data else "training"

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

            model.eval()

            # Reserve a dictionary to store the various results in
            perf_measures = {"loss": -1, "total": -1, "accuracy": -1,
                             "macro_f1_score": -1, "macro_f1_score_main": -1, 'f1_scores': None}
            performance = {"epoch": epoch, "iteration": iteration,
                           "training": perf_measures,
                           "validation": perf_measures.copy()}

            # Obtain predictions and loss on training data; predictions are in order.
            train_predictions, train_mean_loss = get_predictions_and_mean_loss(model,
                                                                               inputs,
                                                                               targets,
                                                                               indices[0:100],
                                                                               batch_size=settings.batch_size,
                                                                               loss_function=loss_function)

            # Get all scores and insert them into the dictionary
            train_scores = get_scores(train_predictions, targets)
            performance["training"].update(train_scores)
            performance["training"]["loss"] = train_mean_loss

            # Also keep track of the relative performance increase/decrease:
            f1_diff_train = performance["training"]["macro_f1_score"] - prev_train_score
            prev_train_score = performance["training"]["macro_f1_score"]

            # To avoid None error during training if without crossvalidation (no validation data).
            val_mean_loss = train_mean_loss

            # And do the same for validation data:
            if use_validation_data:
                val_predictions, val_mean_loss = get_predictions_and_mean_loss(model,
                                                                               val_inputs,
                                                                               val_targets,
                                                                               batch_size=settings.batch_size,
                                                                               loss_function=loss_function)

                val_scores = get_scores(val_predictions, val_targets)
                performance["validation"].update(val_scores)
                performance["validation"]["loss"] = val_mean_loss

            f1_diff_val = performance[stop_criterion_data]["macro_f1_score"] - prev_val_score
            prev_val_score = performance[stop_criterion_data]["macro_f1_score"]

            # Print the various scores
            logger.say('Mean loss: \n  training: {0:12.7f}\n  validation: {1:10.7f}'.format(train_mean_loss, val_mean_loss))
            logger.say('Accuracy: \n  training: {0[training][accuracy]:12.4f} (total {0[training][total]:7d})\n  validation: {0[validation][accuracy]:10.4f} (total {0[validation][total]:7d})'.format(performance))
            logger.say('Macro F1: \n  training: {0[training][macro_f1_score]:12.4f}   dif.: {1:.5f}   (total {0[training][total]:7d})\n  validation: {0[validation][macro_f1_score]:10.4f}   dif.: {2:.5f}   (total {0[validation][total]:7d})\n'.format(performance, f1_diff_train, f1_diff_val))

            # Keep track of best performance, and assess whether to stop training
            if (val_mean_loss < min_val_loss or
                    (performance[stop_criterion_data]["macro_f1_score"] > max_val_macro_f1) or
                    (performance['training']["macro_f1_score"] > max_train_macro_f1)):    # i.e., if the model is improving.
                num_epochs_no_improve = 0
                best_model = model
                min_val_loss = min(val_mean_loss, min_val_loss)
                max_val_macro_f1 = max(performance[stop_criterion_data]["macro_f1_score"], max_val_macro_f1)
                max_train_macro_f1 = max(performance['training']["macro_f1_score"], max_train_macro_f1)
            else:                               # i.e., if the model is NOT improving.
                num_epochs_no_improve += settings.test_every

            if num_epochs_no_improve >= settings.stop_criterion or np.isnan(val_mean_loss):
                message = 'Stopped after epoch {0} because validation loss nor train/validation performance improved for {1} epochs, or validation loss is Nan.'.format(
                    epoch, num_epochs_no_improve)
                logger.say(message)
                logger.log('# ' + message)
                break

        logger.log(performance)

        epoch += 1

    return model, best_model


def get_predictions_and_mean_loss(model, inputs, targets, indices, batch_size=100, loss_function=None, shuffle=True):

    model.eval()
    predictions = []
    weighted_loss = 0
    total_weight = 0

    for batch_indices in data_utils.batches(indices, batch_size, shuffle=shuffle):

        model.init_hidden(len(batch_indices))

        with torch.no_grad():

            batch_outputs = model.generate(batch_indices, predict_len=20)
            preds = [pred[1:].split('<')[0] for pred in batch_outputs]

            print(preds)

        if loss_function is not None:
            # weight = batch_targets.nelement()
            # loss = loss_function(batch_outputs.view(-1, batch_outputs.size()[-1]), batch_targets_var.view(-1))
            # loss = loss.item()
            loss, weight = 1, 1    # TODO
            weighted_loss += weight * loss
            total_weight += weight

        predictions.extend(preds)

    if loss_function is None:
        return predictions

    avg_loss = weighted_loss / total_weight

    return predictions, avg_loss


def get_scores(predictions, targets):
    """
    Computes prediction accuracy, F1 scores for all/main entities, macro average of F1 scores for all/main entities.
    :param predictions: list of predicted labels
    :param targets: list of true labels
    :param restrict_indices: numpy array containing indices (if any) to restrict the scores to (e.g., only pronouns).
    :return: dictionary of all scores
    """
    # TODO @Future: More of this could be done on gpu (e.g.: https://www.kaggle.com/igormq/f-beta-score-for-pytorch/code )

    all_scores = {}

    # Return appropriately named scores:
    all_scores.update({'accuracy': 0,
                       'macro_f1_score': 0,
                       'total': len(targets),
                       })

    return all_scores
