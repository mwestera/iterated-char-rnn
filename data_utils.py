import numpy as np
import string
from matplotlib import pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import torch

import distance
import math

import os
import pickle
import pandas as pd

import random


all_characters = '>' + '<' + string.ascii_lowercase
n_characters = len(all_characters)


vowels = "aeiou"
consonants = "".join(set(string.ascii_lowercase) - set(vowels))


def _sort_sequence(y):
    """
    Relabels clusters so that new occurrences are labeled incrementally.
    Example: [2,1,2,0,1,0] becomes [0,1,0,2,1,2]
    :param y: a numpy array of ints
    :return: a numpy array of ints
    """
    label_map = {}    # a mapping from old to new labels
    y_sorted = np.zeros(len(y), dtype=np.int)   # a new sequence
    next_new_label = 0     # the first unmapped old label we encounter will be mapped to 0
    for i, c in enumerate(y):
        if c not in label_map:
            label_map[c] = next_new_label
            next_new_label += 1
        y_sorted[i] = label_map[c]
    return y_sorted


def _add_delay(X, y, delay, num_features):
    """
    Lengthens sequences X and y by adding nonsense at the start of y and nonsense at the end of X.
    (The loss function will know to ignore the -1 values in y.)
    :param X: Numpy array of points, sequence of points to be clustered.
    :param y: Numpy array of ints, sequence of cluster labels.
    :param delay: int, the LSTM will output the cluster label for the (n)th point at the (n+delay)th timestep.
    :return: Numpy arrays X, y:
        - X is sequence of points, with delay number of -1's at the end.
        - y is sequence of ints (cluster labels), with delay number of -1's at the start.
    """
    X = np.concatenate((X,-1*np.ones((delay,num_features))))
    y = np.concatenate((-1*np.ones(delay,dtype=int),y))
    return X,y


def create_data_total_random(num_sequences, seq_len, delay, centers, num_features, cluster_stddev, plot_clusterings, k_means):
    """
    Creates two arrays of sequences, one of points and one of cluster labels.
    :param num_sequences: int, number of sequences returned
    :param seq_len: int, length of each sequence
    :param delay: int, the LSTM will output the cluster label for the (n)th point at the (n+delay)th timestep.
    :return: two numpy arrays X,y wrapped in tensors wrapped in variables.
        - X (seq_len x num_sequences x num_features); contains sequences of points.
        - y (seq_len x num_sequences); contains sequences of cluster labels.
    """

    # For both X and y, dimension 0 is the position in the sequence; sequences are stacked in dimension 1.
    # This is because that's how PyTorch LSTMs want their inputs.
    X, y = None, None
    for i in range(num_sequences):
        # Create another sequence X and y
        X2, y2 = make_blobs(n_samples=seq_len, centers=centers, n_features=num_features, cluster_std=cluster_stddev,
                          center_box=(-1, 1))
        X2 = X2.astype(np.float32)

        # Optionally plot the first few created clusterings to get an impression:
        if i < plot_clusterings:
            plt.scatter(X2[:,0], X2[:,1], marker='o', c=y2, s=25, edgecolor='k')
            plt.show()

        # Sort and optionally delay the sequence
        y2 = _sort_sequence(y2)
        X2, y2 = _add_delay(X2, y2, delay, num_features)

        # Add a dimension at the start along which sequences will be stacked (batch).
        X2 = X2.reshape(1, seq_len+delay, num_features)
        y2 = y2.reshape(1, seq_len+delay)

        # Stack this sequence to the previous ones (if any).
        if i == 0:
            X, y = X2, y2
        else:
            X = np.concatenate((X,X2), 0)
            y = np.concatenate((y,y2), 0)

    # Optional: For comparison, test and print the performance of a standard K-means clusterer on validation set 1
    if k_means:
        print('Computing K-means baseline on X_val1...')
        kmeans_accuracy = compute_k_means(X, y, centers if isinstance(centers, int) else len(centers))
        print('K-means accuracy on validation set 1: {0}%'.format(kmeans_accuracy))

    # X now contains the desired amount of sequences, and y the corresponding cluster labels.
    return {'X':X,'y':y}


def create_data_in_domains(settings, compute_k_means=False):

    num_domains = settings.num_domains
    num_clusters_per_domain = settings.clusters_per_domain
    num_features = settings.num_features
    seq_per_domain = settings.seq_per_domain
    cluster_stddev = settings.cluster_stddev
    seq_len = settings.seq_len
    delay = settings.delay

    domains = np.random.uniform(-1.0, 1.0, (num_domains, num_clusters_per_domain, num_features))
    data = []
    for domain in domains:
        sequences = create_data_total_random(seq_per_domain, seq_len, delay, domain, num_features, cluster_stddev, 0, compute_k_means)
        data.append(sequences)
    return data


def random_word(min_len, max_len):
    length = int(round(random.betavariate(3,8) * (max_len - min_len) + min_len))
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(consonants)
        else:
            word += random.choice(vowels)

    return word


def get_data(settings, stats=True):

    if os.path.exists(settings.dataset):

        pretrained = np.genfromtxt(settings.dataset, delimiter=' ', dtype=str, invalid_raise=False,
                                   max_rows=settings.max_words)
        idx_to_word = pretrained[:, 0]
        word_vectors = pretrained[:, 1:].astype(float)

    else:

        datapath = os.path.join('data', settings.dataset + '.txt')

        num_words = settings.max_words
        idx_to_word = np.array([random_word(settings.min_len, settings.max_len) for _ in range(num_words)])
        word_vectors = 2 * np.random.rand(num_words, settings.num_dims) - 1      # uniform in -1,1
        sigmoid = lambda x: 1 / (1 + np.exp(-3 * x))       # 3 controls how polarizing it is.  # TODO better distribution? Blobs perhaps?
        word_vectors = 2 * sigmoid(word_vectors) - 1

        with open(datapath, 'w') as file:
            for i in range(num_words):
                file.write(idx_to_word[i] + ' ' + ' '.join(['{0:.5f}'.format(v) for v in word_vectors[i]]) + '\n')


    suitable_indices = np.array([i for i in range(len(idx_to_word)) if idx_to_word[i].isalpha()])

    print('Loaded word embeddings ({0} words; {1} suitable).'.format(len(idx_to_word), len(suitable_indices)))

    idx_to_word = idx_to_word[suitable_indices]
    word_vectors = word_vectors[suitable_indices]

    word_to_idx = {idx_to_word[i]: i for i in range(len(idx_to_word))}

    if stats:
        avg_inter_str_dist, avg_inter_sem_dist, comp = correlation(word_vectors, idx_to_word, idx_to_word[:500],
                                                               word_ids=list(range(500)))
        print('correlation of top 1000 words:', avg_inter_str_dist, avg_inter_sem_dist, comp)
        # TODO Expand this.

    return idx_to_word, word_to_idx, word_vectors


def to_char_tensors(idx_to_word):
    word_ids = torch.arange(len(idx_to_word)).int()
    max_len = max([len(idx_to_word[w]) for w in word_ids]) + 2
    char_tensors = torch.LongTensor(len(idx_to_word), max_len)
    for i, word_idx in enumerate(word_ids):
        char_tensors[i] = char_tensor(idx_to_word[word_idx], padding=max_len)
    inputs = char_tensors[:, :-1]
    inputs[inputs == -1] = 1  # To avoid error
    targets = char_tensors[:, 1:]

    return inputs, targets


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


def correlation(word_vectors, idx_to_word, strings=None, word_ids=None):
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



def compute_k_means(inputs, targets, num_clusters):
    seq_len = inputs.shape[1]
    inputs = np.concatenate(inputs)
    targets = np.concatenate(targets)
    predictions = KMeans(n_clusters=num_clusters).fit_predict(inputs)
    for i in range(0, len(inputs), seq_len):
        predictions[i:i+seq_len] = _sort_sequence(predictions[i:i+seq_len])
    Kmeans_accuracies = 100 * np.average(targets == predictions)
    return np.average(Kmeans_accuracies)


def batches(indices, batch_size, shuffle=True):

    if shuffle:
        indices = indices.copy()
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        yield indices[i:i+batch_size]


def read_answers_csv(path, logger=None):
    preds = pd.read_csv(path, delim_whitespace=True, index_col=[0,1], comment='#', header=None, names=['dom','seq']+list(range(0,30))) #, dtype={'index': int, 'prediction': int, 'target': int})
    # TODO avoid hard-coding of sequence length
    targs = preds.copy()
    for i in range(0, len(preds.columns)):
        preds[i] = preds[i].apply(lambda s: int(s.split(':')[0]))
        targs[i] = targs[i].apply(lambda s: int(s.split(':')[1]))
    catted = pd.concat([preds, targs], keys=['predictions', 'targets'])
    if logger is not None:
        logger.whisper('Predictions (and targets and indices) read from '+path)
    return catted