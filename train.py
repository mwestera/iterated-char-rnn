#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import distance

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import pickle
import numpy as np

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=10000)
argparser.add_argument('--print_every', type=int, default=50)
argparser.add_argument('--hidden_size', type=int, default=400)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--batch_size', type=int, default=100)
# argparser.add_argument('--gen_temp', type=int, default=.1)
argparser.add_argument('--max_words', type=int, default=10000)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--no_cuda', action='store_true')
args = argparser.parse_args()

# TODO Implement config_utils for random hyperparameter search on cluster.
# TODO Add hyperparameter to feed word embedding at every decoder step.
# TODO I can test bootstrapping hypotheses, slowly increasing set of words, ...

if not args.no_cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)


def random_training_set(training_ids, batch_size):
    word_ids = torch.LongTensor(random.sample(training_ids, batch_size))
    max_len = max([len(idx_to_word[w]) for w in word_ids]) + 2
    char_tensors = torch.LongTensor(batch_size, max_len)
    for i, word_idx in enumerate(word_ids):
        char_tensors[i] = char_tensor(idx_to_word[word_idx], padding=max_len)
    inp = char_tensors[:,:-1]
    inp[inp==-1] = 1        # To avoid error
    target = char_tensors[:,1:]

    inp = Variable(inp)
    target = Variable(target)
    word_ids = Variable(word_ids)
    if not args.no_cuda:
        inp = inp.cuda()
        target = target.cuda()
        word_ids = word_ids.cuda()

    return word_ids, inp, target


def train(word_ids, inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if not args.no_cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    max_len = inp.size()[1]

    for c in range(max_len):
        output, hidden = decoder(inp[:,c], hidden, seed=word_ids if c == 0 else None)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / max_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def correlation(word_vectors, strings=None, word_ids=None):
    if word_ids is None:
        word_ids = list(range(len(word_vectors)))
        if strings is None:
            strings = [idx_to_word[i] for i in word_ids]

    n_words = len(word_ids)

    inter_str_dist = np.zeros((n_words, n_words)) - 1
    inter_sem_dist = np.zeros((n_words, n_words)) - 1
    for i in range(n_words):
        for j in range(n_words):
            if i < j:
                inter_str_dist[i, j] = distance.levenshtein(strings[i], strings[j]) / max(len(strings[i]), len(strings[j]))
                inter_sem_dist[i, j] = 1.0 - torch.nn.functional.cosine_similarity(word_vectors[word_ids[i]],
                                                                                   word_vectors[word_ids[j]],
                                                                                   dim=0).item()
    avg_inter_sem_dist = np.average(inter_sem_dist[inter_sem_dist >= 0])
    avg_inter_str_dist = np.average(inter_str_dist[inter_str_dist >= 0])

    sum_difs_sem_str = 0
    sum_sq_difs_sem = 0
    sum_sq_difs_str = 0
    for i in range(n_words):
        for j in range(n_words):
            if i < j:
                sum_difs_sem_str += (inter_sem_dist[i, j] - avg_inter_sem_dist) * (
                        inter_str_dist[i, j] - avg_inter_str_dist)
                sum_sq_difs_sem += (inter_sem_dist[i, j] - avg_inter_sem_dist) ** 2
                sum_sq_difs_str += (inter_str_dist[i, j] - avg_inter_str_dist) ** 2

    comp = sum_difs_sem_str / math.sqrt(sum_sq_difs_sem * sum_sq_difs_str)

    return avg_inter_str_dist, avg_inter_sem_dist, comp


# word_vectors = np.load('friends.train.scene_delim__GoogleNews-vectors-negative300.npy')[1:]
pretrained = np.genfromtxt('data/glove.6B/glove.6B.50d.txt', delimiter=' ', dtype=str, invalid_raise=False, max_rows=args.max_words)
idx_to_word = pretrained[:,0]
word_to_idx = {idx_to_word[i]: i for i in range(len(idx_to_word))}
word_vectors = pretrained[:,1:].astype(float)

suitable_indices = [i for i in range(len(idx_to_word)) if idx_to_word[i].isalpha()]

print('Loaded word embeddings ({0} words; {1} suitable).'.format(len(idx_to_word), len(suitable_indices)))

# Initialize models and start training
decoder = CharRNN(
    torch.Tensor(word_vectors),
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,decoder.parameters()), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

if not args.no_cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

avg_inter_str_dist, avg_inter_sem_dist, comp = correlation(decoder.seed.weight, idx_to_word[:500], word_ids=list(range(500)))
print('correlation of top 1000 words:', avg_inter_str_dist, avg_inter_sem_dist, comp)

try:
    print("Training for %d epochs..." % args.n_epochs)

    training_inds = suitable_indices        # TODO implement bottleneck

    for epoch in range(1, args.n_epochs + 1):
        # TODO epochs != iterations
        # TODO make sure ALL training_inds are covered (to avoid inevitable learning bottleneck).
        loss = train(*random_training_set(training_inds, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:

            preview_inds = training_inds[0:100]

            # TODO Go through preview inds in batches in case it's too many.
            predictions = generate(decoder, preview_inds, predict_len=20, cuda=not args.no_cuda)        # temperature=args.gen_temp,
            predictions = [pred[1:].split('<')[0] for pred in predictions]
            str_sims = []
            sem_sims = []
            com_accs = []

            for i, pred in zip(preview_inds, predictions):
                str_sim = 1.0 - (distance.levenshtein(idx_to_word[i], pred) / max(len(idx_to_word[i]), len(pred)))

                if str_sim == 1:
                    sem_sim = 1
                else:
                    j = word_to_idx.get(pred)
                    if j is None:
                        sem_sim = 0
                    else:
                        sem_sim = torch.nn.functional.cosine_similarity(decoder.seed.weight[i], decoder.seed.weight[j], dim=0).item()

                com_acc = max(str_sim, sem_sim)     # TODO Make smarter, e.g., semantic closeness of closest string(s).

                str_sims.append(str_sim)
                sem_sims.append(sem_sim)
                com_accs.append(com_acc)


                # if sem_sim < 1.0 and sem_sim > 0.0:
                #     print('{0} {1} ({2:.2f}, {3:.2f})'.format(idx_to_word[i], pred, str_sim, sem_sim))

            avg_str_sim = sum(str_sims) / len(str_sims)
            avg_sem_sim = sum(sem_sims) / len(sem_sims)
            avg_com_acc = sum(com_accs) / len(com_accs)

            avg_inter_str_dist, avg_inter_sem_dist, comp = correlation(decoder.seed.weight, predictions, preview_inds)  # TODO not feasible to do for all words; reuse the semantic one.

            print('[{0} ({1} {2:.1f}%) {3:.4f}  [{4:.2f} {5:.2f} {6:.2f}]  [{7:.2f} {8:.2f} {9:.2f}]]'.format(
                time_since(start), epoch, epoch / args.n_epochs * 100, loss,
                avg_str_sim, avg_sem_sim, avg_com_acc,
                avg_inter_str_dist, avg_inter_sem_dist, comp))

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

