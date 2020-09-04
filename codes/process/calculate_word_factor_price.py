"""
calculate_word_factor_price.py
@date: 2020/09/01
"""

import math
import string
import pickle
import enchant
import pandas as pd
from statistics import median
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict

all_stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
english_dict = enchant.Dict("en_US")

def remove_digits(s):
    return ''.join([i for i in s if not i.isdigit()])

def lemmatize(s):
    return wordnet_lemmatizer.lemmatize(s.lower().encode("ascii", "ignore").decode("utf-8"))

data_path = "../../data/"
menu_fname = "menu_price_1125.csv"
menu_frame = pd.read_csv(data_path + menu_fname, na_values="None")

# read vocab
vocab = pickle.load( open("../../params/menu_vocab_v2.p", "rb"))

word_factor_prices = {}
word_factor_count = defaultdict(int)
word_factor_price_avg = {}
word_factor_price_med = {}

for index, df in menu_frame.iterrows():

    if index % 10000 == 0: print ("index:", index)
    item = df["item"]
    factor = df["city"]

    try:

        if not math.isnan(df["price"]):

            tmp_words = item.lower().split(" ")
            tmp_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in tmp_words]
            tmp_words = [remove_digits(word) for word in tmp_words]
            tmp_words = [lemmatize(word) for word in tmp_words]
            tmp_words = [word for word in tmp_words if len(word) > 1]
            tmp_words = [word for word in tmp_words if english_dict.check(word)]
            tmp_words = [word.strip() for word in tmp_words]

            for word in tmp_words:
                if word not in word_factor_prices:
                    word_factor_prices[(word, factor)] = [df["price"]]
                else:
                    word_factor_prices[(word, factor)].append(df["price"])
                word_factor_count[(word, factor)] += 1

    except AttributeError:
        pass

for word, factor in word_factor_prices:
    # word_factor_price_avg[(word, factor)] = sum(word_factor_prices[(word, factor)]) / word_factor_count[(word, factor)]
    word_factor_price_med[(word, factor)] = median(word_factor_prices[(word, factor)])

pickle.dump(word_factor_price_avg, open("menu_word_factor_avg_price.p", "wb"))
pickle.dump(word_factor_price_med, open("menu_word_factor_med_price.p", "wb"))
pickle.dump(word_factor_prices, open("menu_word_factor_prices.p", "wb"))
