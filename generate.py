#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *

def generate(decoder, word_ids, prime_str='>', predict_len=20, cuda=False):

    n_words = len(word_ids)

    predicted = [prime_str for _ in range(n_words)]

    inp = torch.zeros(n_words, 1).long()
    for i in range(n_words):
        inp[i] = char_tensor(prime_str)

    inp = Variable(inp)
    hidden = decoder.init_hidden(n_words)
    word_ids = Variable(torch.LongTensor(word_ids))
    if cuda:
        hidden = hidden.cuda()
        inp = inp.cuda()
        word_ids = word_ids.cuda()


    for p in range(predict_len):

        output, hidden = decoder(inp, hidden, seed=word_ids if p == 0 else None)

        # Sample from the network as a multinomial distribution
        # output_dist = output.data.view(n_words, -1).div(temperature).exp()

        values, top_i = torch.max(output, 1)

        # Add predicted character to string and use as next input
        predicted_char = [all_characters[i] for i in top_i]
        predicted = [predicted[i] + predicted_char[i] for i in range(n_words)]

        inp = torch.zeros(n_words, 1).long()
        for i in range(n_words):
            inp[i] = char_tensor(predicted_char[i])
        inp = Variable(inp)
        if cuda:
            inp = inp.cuda()

    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))

