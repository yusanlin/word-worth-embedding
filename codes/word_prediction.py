"""
word_prediction.py
"""

import os
import math
import string
import pickle
import enchant
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter, defaultdict

from scipy.spatial import KDTree

hidden_dim = 128

all_stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
english_dict = enchant.Dict("en_US")


def remove_digits(s):
    return ''.join([i for i in s if not i.isdigit()])


def lemmatize(s):
    return wordnet_lemmatizer.lemmatize(s.lower().encode("ascii", "ignore").decode("utf-8"))


def clean(s):
    tmp_words = s.lower().split(" ")
    tmp_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in tmp_words]
    tmp_words = [remove_digits(word) for word in tmp_words]
    tmp_words = [lemmatize(word) for word in tmp_words]
    tmp_words = [word for word in tmp_words if len(word) > 1]
    tmp_words = [word for word in tmp_words if english_dict.check(word)]
    tmp_words = [word.strip() for word in tmp_words]

    return tmp_words

def read_menu(data_path, menu_fname, na_values="None", encoding="utf-8"):
    """
    parameters
    - encoding: the file encoding of menu_fname. Options: utf-8, ISO-8859-1
    """
    return  pd.read_csv(os.path.join(data_path, menu_fname), na_values=na_values, encoding=encoding)

def precision(y_pred, y_true, k):
    y_pred_tmp = y_pred[:k+1]
    overlap = set(y_pred_tmp).intersection(set(y_true))
    return float(len(overlap)) / k

def recall(y_pred, y_true, k):
    y_pred_tmp = y_pred[:k+1]
    overlap = set(y_pred_tmp).intersection(set(y_true))
    return float(len(overlap)) / len(y_true)

for dataset in ["shoe", "retail", "reward"]:

    if dataset == "retail":
        menu_fname = "retail_price.csv"
        vocab_fname = "retail_vocab_v2.p"
        encoding = "ISO-8859-1"
        item_column_name = "Description"
        factor_column_name = "Country"
        price_column_name = "UnitPrice"

    elif dataset == "menu":
        menu_fname = "menu_price_1125.csv"
        vocab_fname = "menu_vocab_v2.p"
        encoding = "utf-8"
        item_column_name = "item"
        factor_column_name = "city"
        price_column_name = "price"

    elif dataset == "shoe":
        menu_fname = "shoe_price.csv"
        vocab_fname = "shoe_vocab_v2.p"
        encoding = "ISO-8859-1"
        item_column_name = "name"
        factor_column_name = "brand"
        price_column_name = "prices.amountMax"

    elif dataset == "reward":
        menu_fname = "reward_price.csv"
        vocab_fname = "reward_vocab_v2.p"
        encoding = "utf-8"
        item_column_name = "description"
        factor_column_name = "category"
        price_column_name = "price"

    menu_frame = read_menu("../data/", menu_fname, encoding=encoding)

    X_dict = pickle.load(open("../trained_parameters/wp_" + dataset + "_X.p", "rb"))
    all_words = [word for word in X_dict.keys()]

    questions = {}

    for index, df in menu_frame.iterrows():

        try:
            words = clean(df[item_column_name])
            price = df[price_column_name]

            if not math.isnan(price) and all([word in all_words for word in words]) and len(words) > 1:
                if " ".join(words[:-1]) not in questions:
                    questions[" ".join(words[:-1])] = [words[-1]]
                else:
                    questions[" ".join(words[:-1])].append(words[-1])

        except AttributeError:
            pass

    for query in questions:
        questions[query] = list(set(questions[query]))

    for model_name in ["wp", "word2vec", "mwp", "mwcp", "joint_wwm"]:

        X_dict = pickle.load(open("../trained_parameters/"+ model_name + "_" + dataset + "_X.p", "rb"))

        all_words = [word for word in X_dict.keys()]

        X = []

        for query in questions:
            tmp = np.zeros(hidden_dim)
            words = query.split(" ")
            for word in words:
                tmp += X_dict[word]
            X.append(tmp)

        X = np.asarray(X)

        embedding = []

        for word in X_dict:
            x = X_dict[word]
            embedding.append(x)

        embedding = np.asarray(embedding)

        tree = KDTree(embedding)

        precisions_sum = defaultdict(float)
        recalls_sum = defaultdict(float)
        queries = [q for q in questions.keys()]

        for i in range(X.shape[0]):

        #     if i % 1000 == 0: print (i)

            query = queries[i]
            query_x = X[i]

            k = 1000 # maximum number of words retrived
            nearest_indices = tree.query(query_x, k+1)[1]
            nearest_words = []
            for j in nearest_indices:
                try:
                    nearest_words.append(all_words[j])
                except IndexError:
                    pass

            for l in range(100, k+1, 100):

                precisions_sum[l] += precision(nearest_words, questions[query], l)
                recalls_sum[l]    += recall(nearest_words, questions[query], l)

        for l in range(100, k+1, 100):
            precisions_sum[l] /= X.shape[0]
            recalls_sum[l] /= X.shape[0]

        for l in range(100, k+1, 100):
            print (dataset, model_name, "precision", l, precisions_sum[l])
            print (dataset, model_name, "recall", l, recalls_sum[l])
