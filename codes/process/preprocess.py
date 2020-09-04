"""
preprocess.py
"""

import os
import math
import string
import pickle
import enchant
import pandas as pd

from statistics import median
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter, defaultdict

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


def create_vocab(menu_frame, item_column_name, n_words_vocab, vocab_fname):
    words = []

    for item in menu_frame[item_column_name]:
        try:
            tmp_words = clean(item)
            words.extend(tmp_words)
        except AttributeError:
            pass

    word_counter = Counter(words)

    rank = 0
    i = 0

    word_counter_populated = word_counter.most_common()

    vocab = []

    while rank < n_words_vocab:
        word, count = word_counter_populated[i]
        if word not in all_stopwords and word != "":
            print (word, count)
            vocab.append(word)
            rank += 1
        i += 1

    pickle.dump(vocab, open(vocab_fname, "wb"))

def calculate_word_avg_price(menu_frame, item_column_name, factor_column_name, price_column_name, prices_fname, avg_price_fname, factor_avg_price_fname, factor_prices_fname, factor_med_price_fname):
    word_prices = {}
    word_count = defaultdict(int)
    word_price_avg = {}

    word_factor_prices = {}
    word_factor_count = defaultdict(int)
    word_factor_price_avg = {}
    word_factor_price_med = {}

    for index, df in menu_frame.iterrows():

        if index % 10000 == 0: print ("index:", index)
        item = df[item_column_name]
        factor = df[factor_column_name]

        try:

            if not math.isnan(df[price_column_name]):
                tmp_words = clean(item)

                for word in tmp_words:
                    if word not in word_prices:
                        word_prices[word] = [df[price_column_name]]
                    else:
                        word_prices[word].append(df[price_column_name])

                    if (word, factor) not in word_factor_prices:
                        word_factor_prices[(word, factor)] = [df[price_column_name]]
                    else:
                        word_factor_prices[(word, factor)].append(df[price_column_name])

                    word_count[word] += 1
                    word_factor_count[(word, factor)] += 1
        except AttributeError:
            pass

    for word in word_prices:
        word_price_avg[word] = sum(word_prices[word]) / word_count[word]

    for word, factor in word_factor_prices:
        word_factor_price_avg[(word, factor)] = sum(word_factor_prices[(word, factor)]) / word_factor_count[(word, factor)]
        word_factor_price_med[(word, factor)] = median(word_factor_prices[(word, factor)])

    pickle.dump(word_price_avg, open(avg_price_fname, "wb"))
    pickle.dump(word_prices, open(prices_fname, "wb"))
    pickle.dump(word_factor_price_avg, open(factor_avg_price_fname, "wb"))
    pickle.dump(word_factor_prices, open(factor_prices_fname, "wb"))
    pickle.dump(word_factor_price_med, open(factor_med_price_fname, "wb"))

menu_frame_reward = read_menu(data_path="../../data/", menu_fname="shoe_price.csv", na_values="None", encoding="ISO-8859-1")
# create_vocab(menu_frame_reward, item_column_name="description", n_words_vocab=1000, vocab_fname="reward_vocab_v2.p")
calculate_word_avg_price(menu_frame=menu_frame_reward,
                                                        item_column_name="name",
                                                        factor_column_name="brand",
                                                        price_column_name="prices.amountMax",
                                                        prices_fname="shoe_word_prices.p",
                                                        avg_price_fname="shoe_word_avg_price.p",
                                                        factor_avg_price_fname="shoe_factor_word_avg_price.p",
                                                        factor_prices_fname="shoe_factor_word_prices.p",
                                                        factor_med_price_fname="shoe_factor_word_med_price.p")
