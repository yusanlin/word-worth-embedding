import os
import math
import torch
import pickle
import string
import enchant
import random

import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

all_stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
english_dict = enchant.Dict("en_US")

# -------------
# Functions
# -------------

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0

# def clean(word):
#     try:
#         tmp = wordnet_lemmatizer.lemmatize(word.lower().encode("ascii", "ignore").decode("utf-8"))
#         return tmp
#     except UnicodeDecodeError:
#         return ""

def remove_digits(s):
    return ''.join([i for i in s if not i.isdigit()])

def lemmatize(s):
    return wordnet_lemmatizer.lemmatize(s.lower().encode("ascii", "ignore").decode("utf-8"))

# ----------
# Classes
# ----------

class MenuDataset(Dataset):
    """
    Menu Dataset
    """

    data_path = "../data/"
    param_path = "../params/"

    def __init__(self, menu_fname, vocab_fname, sampling="wp", beta=None, encoding="utf-8", item_column_name="item", factor_column_name="factor", price_column_name="price"):

        self.menu_fname = menu_fname
        self.vocab_fname = vocab_fname
        self.sampling = sampling

        # when reading in retail and shoe dataset, set encoding="ISO-8859-1"
        self.menu_frame = pd.read_csv(self.data_path + self.menu_fname, na_values="None", encoding=encoding)
        self.menu_frame.dropna(axis=0, how='any', inplace=True)
        self.top_words = pickle.load(open(self.param_path + self.vocab_fname, "rb"))
        self.top_words = ["<PAD>"] + self.top_words
        self.n_vocab = len(self.top_words)

        self.beta = beta

        self.vocab = {}
        for i in range(self.n_vocab):
            word = self.top_words[i]
            self.vocab[word] = i

        # process the words
        for index, row in self.menu_frame.iterrows():
            # TODO: need to do more cleaning here, follow the new version of vocab generation
            # words = [clean(word) for word in row["item"].split(" ")]

            item = row[item_column_name]

            tmp_words = item.lower().split(" ")
            tmp_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in tmp_words]
            tmp_words = [remove_digits(word) for word in tmp_words]
            tmp_words = [lemmatize(word) for word in tmp_words]
            tmp_words = [word for word in tmp_words if len(word) > 1]
            tmp_words = [word for word in tmp_words if english_dict.check(word)]
            tmp_words = [word.strip() for word in tmp_words]

            words = [word for word in tmp_words if word in self.top_words]

            self.menu_frame.at[index, item_column_name] = words

        # remove those with only one word in the item
        self.menu_frame = self.menu_frame[self.menu_frame[item_column_name].map(len) > 1]

        self.max_word_length = 10
        self.max_yp = self.menu_frame[price_column_name].max()
        self.all_factors = list(set(self.menu_frame.iloc[:, 0]))
        self.all_factors.sort()
        self.n_factors = len(self.all_factors)

    def __len__(self):
        return len(self.menu_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.sampling == "wp":

            factor = self.menu_frame.iloc[idx, 0]
            words = self.menu_frame.iloc[idx, 1]
            price = self.menu_frame.iloc[idx, 2]

            x_factor = np.zeros(self.n_factors)
            x_words = np.zeros(self.n_vocab)
            y_price = np.zeros(1)

            x_factor[self.all_factors.index(factor)] = 1

            for word in words:
                x_words[self.top_words.index(word)] += 1

            y_price[0] = sigmoid(price)

            x_factor  = torch.from_numpy(x_factor).float()
            x_words = torch.from_numpy(x_words).float()
            y_price = torch.from_numpy(y_price).float()

            sample = {"factor": x_factor, "words": x_words, "price": y_price}

        elif self.sampling == "word2vec":

            words = self.menu_frame.iloc[idx, 1]

            x_target_word = np.zeros(1)
            x_context_words = np.zeros(self.n_vocab)

            i_target_word = int(np.floor(len(words) / 2.0 ))

            x_target_word[0] = self.top_words.index(words[i_target_word])

            for i in range(len(words)):

                if i != i_target_word:
                    x_context_words[i] += 1.0

            x_context_words[i] /= (len(words)-1)

            x_target_word = torch.from_numpy(x_target_word).long()
            x_context_words = torch.from_numpy(x_context_words).float()

            sample = {"target_word": x_target_word, "context_words": x_context_words}

        elif self.sampling == "mwp":

            words = self.menu_frame.iloc[idx, 1]
            price = self.menu_frame.iloc[idx, 2]

            x_target_word = np.zeros(1)
            x_context_words = np.zeros(self.n_vocab)
            y_price = np.zeros(1)

            i_target_word = int(np.floor(len(words) / 2.0 ))
            x_target_word[0] = self.top_words.index(words[i_target_word])

            for i in range(len(words)):

                if i != i_target_word:
                    x_context_words[i] += 1.0

            x_context_words[i] /= (len(words)-1)

            y_price[0] = sigmoid(price)

            x_target_word = torch.from_numpy(x_target_word).long()
            x_context_words = torch.from_numpy(x_context_words).float()
            y_price = torch.from_numpy(y_price).float()

            sample = {"target_word": x_target_word, "context_words": x_context_words, "price": y_price}

        elif self.sampling == "mwcp":

            factor = self.menu_frame.iloc[idx, 0]
            words = self.menu_frame.iloc[idx, 1]
            price = self.menu_frame.iloc[idx, 2]

            x_factor = np.zeros(self.n_factors)
            x_target_word = np.zeros(1)
            x_context_words = np.zeros(self.n_vocab)
            y_price = np.zeros(1)

            x_factor[self.all_factors.index(factor)] = 1

            i_target_word = int(np.floor(len(words) / 2.0 ))
            x_target_word[0] = self.top_words.index(words[i_target_word])

            for i in range(len(words)):

                if i != i_target_word:
                    x_context_words[i] += 1.0

            x_context_words[i] /= (len(words)-1)

            y_price[0] = sigmoid(price)

            x_factor  = torch.from_numpy(x_factor).float()
            x_target_word = torch.from_numpy(x_target_word).long()
            x_context_words = torch.from_numpy(x_context_words).float()
            y_price = torch.from_numpy(y_price).float()

            sample = {"factor": x_factor, "target_word": x_target_word, "context_words": x_context_words, "price": y_price}

        elif self.sampling == "wholeseq":

            factor = self.menu_frame.iloc[idx, 0]
            words = self.menu_frame.iloc[idx, 1]
            price = self.menu_frame.iloc[idx, 2]

            length = min(len(words), self.max_word_length)

            x_factor = np.zeros(self.n_factors)
            x_words = np.zeros(self.max_word_length)
            y_price = np.zeros(1)

            x_factor[self.all_factors.index(factor)] = 1

            for i in range(length):
                word = words[i]
                x_words[i] = self.top_words.index(word)

            y_price[0] = sigmoid(price)

            x_factor  = torch.from_numpy(x_factor).float()
            x_words = torch.from_numpy(x_words).long()
            y_price = torch.from_numpy(y_price).float()

            sample = {"factor": x_factor, "words": x_words, "price": y_price, "length": length}

        elif self.sampling == "subseq":

            factor = self.menu_frame.iloc[idx, 0]
            words = self.menu_frame.iloc[idx, 1]
            price = self.menu_frame.iloc[idx, 2]

            length = min(len(words), self.max_word_length)

            x_factor = np.zeros(self.n_factors)
            x_subseq = np.zeros(self.max_word_length)
            x_missing = np.zeros(1)
            y_price = np.zeros(1)

            x_factor[self.all_factors.index(factor)] = 1

            subseq_start = np.random.randint(length)
            subseq_end = subseq_start + int(length * self.beta)+1
            subseq_words = words[subseq_start: subseq_end]
            missing_words = words[:subseq_start] + words[subseq_end:]

            if subseq_words == words:
                subseq_words = subseq_words[:-1]
                missing_words = [words[-1]]

            # print ("words:", words)
            # print ("subseq:", subseq_words)
            # print ("missing:", missing_words)

            for i in range(len(subseq_words)):
                word = subseq_words[i]
                x_subseq[i] = self.top_words.index(word)

            # for i in range(len(missing_words)):
            word = random.choice(missing_words)
            # x_missing[i] = self.top_words.index(word)
            x_missing[0] = self.top_words.index(word)

            y_price[0] = sigmoid(price)

            x_factor = torch.from_numpy(x_factor).float()
            x_subseq = torch.from_numpy(x_subseq).long()
            x_missing = torch.from_numpy(x_missing).long()
            y_price = torch.from_numpy(y_price).float()

            sample = {"factor": x_factor, "subseq": x_subseq, "missing": x_missing, "price": y_price, "length": length}

        return sample

def create_data_loader(dataset, validation_split, batch_size=256):
    # split into train and valid sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # create samplers for training and validation
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # create data loaders for training and validation
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

# biased coin toss function to determine which sampling scheme to use
def flip(alpha):
    return 'wholeseq' if random.random() < alpha else 'subseq'

def process_batch_to_input(sample_batched, sampling, device):

        if sampling =="wholeseq":

            X = sample_batched["words"]
            Xc = sample_batched["factor"]
            y = sample_batched["price"]

            X_lengths = sample_batched["length"]

            # sort the input sequence based on sequence lengths decreasingly
            lengths_sorted, sorted_idx = X_lengths.sort(descending=True)
            X = X[sorted_idx]
            Xc = Xc[sorted_idx]
            y = y[sorted_idx]

            lengths_sorted = lengths_sorted.to(device)
            X = X.to(device)
            Xc = Xc.to(device)
            y = y.to(device)

            data = {"X": X, "Xc": Xc, "y": y, "lengths": lengths_sorted}

        else:
            X_subseq = sample_batched["subseq"]
            X_missing = sample_batched["missing"]
            Xc = sample_batched["factor"]
            y = sample_batched["price"]

            X_lengths = sample_batched["length"]

            lengths_sorted, sorted_idx = X_lengths.sort(descending=True)
            X_subseq = X_subseq[sorted_idx]
            X_missing = X_missing[sorted_idx].squeeze()
            Xc = Xc[sorted_idx]
            y = y[sorted_idx]

            lengths_sorted = lengths_sorted.to(device)
            X_subseq = X_subseq.to(device)
            X_missing = X_missing.to(device)
            Xc = Xc.to(device)
            y = y.to(device)

            data = {"X": X_subseq, "Xc": Xc, "y_price": y, "y_word": X_missing, "lengths": lengths_sorted}

        return data
