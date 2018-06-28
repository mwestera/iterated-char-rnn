#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

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
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=400)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--batch_size', type=int, default=5000)
# argparser.add_argument('--gen_temp', type=int, default=.1)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--no_cuda', action='store_true')
args = argparser.parse_args()

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

# Initialize models and start training
# TODO Also experiment with https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md, which supposedly respect morphological structure more.

# word_vectors = np.load('friends.train.scene_delim__GoogleNews-vectors-negative300.npy')[1:]
pretrained = np.genfromtxt('glove.6B/glove.6B.50d.txt', delimiter=' ', dtype=str, invalid_raise=False)
idx_to_word = pretrained[:,0]
word_to_idx = {i: idx_to_word[i] for i in range(len(idx_to_word))}
word_vectors = pretrained[:,1:].astype(float)

suitable_indices = [i for i in range(len(idx_to_word)) if idx_to_word[i].isalpha()]

print('Loaded word embeddings ({0} words; {1} suitable).'.format(len(idx_to_word), len(suitable_indices)))

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

try:
    print("Training for %d epochs..." % args.n_epochs)

    training_inds = suitable_indices        # TODO implement bottleneck

    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(training_inds, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            preview_inds = training_inds[150:200]
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            predictions = generate(decoder, preview_inds, predict_len=20, cuda=not args.no_cuda)        # temperature=args.gen_temp,
            for i, pred in zip(preview_inds, predictions):
                print(idx_to_word[i], pred[1:].split('<')[0])

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

