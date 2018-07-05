from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from data_utils import char_tensor, all_characters

class CharRNN(nn.Module):
    # Based on https://github.com/spro/char-rnn.pytorch
    def __init__(self, settings, n_characters, seed, logger=None):
        super(CharRNN, self).__init__()
        input_size = n_characters
        hidden_size = settings.hidden_size
        output_size = n_characters
        model = settings.model_type
        n_layers = settings.n_layers

        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.seed = nn.Embedding.from_pretrained(seed, freeze=True)
        # Verify that this thing remains fixed during training.
        self.seed_to_hidden = nn.Linear(seed.shape[1], hidden_size)

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, seed=None):
        if seed is not None:
            seed_emb = self.seed(seed)
            hidden = hidden + self.seed_to_hidden(seed_emb)
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden, seed=None):
        if seed is not None:
            seed_emb = self.seed(seed)
            hidden = hidden + self.seed_to_hidden(seed_emb)
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

    def generate(self, word_ids, prime_str='>', predict_len=20, cuda=True):
        # TODO Obtain CUDA from self.

        n_words = len(word_ids)

        predicted = [prime_str for _ in range(n_words)]

        inp = torch.zeros(n_words, 1).long()
        for i in range(n_words):
            inp[i] = char_tensor(prime_str)

        inp = Variable(inp)
        hidden = self.init_hidden(n_words)
        word_ids = Variable(torch.LongTensor(word_ids))
        if cuda:
            hidden = hidden.cuda()
            inp = inp.cuda()
            word_ids = word_ids.cuda()

        for p in range(predict_len):

            output, hidden = self(inp, hidden, seed=word_ids if p == 0 else None)

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