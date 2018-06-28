# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch

# Reading and un-unicode-encoding data

all_characters = '>' + '<' + string.ascii_letters
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string, padding=False):
    length = padding or len(string)
    tensor = torch.zeros(length).long()
    if padding:
        tensor[0] = 0
        tensor[len(string) + 1] = 1
    for c in range(length):
        try:
            tensor[c if not padding else c+1] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

