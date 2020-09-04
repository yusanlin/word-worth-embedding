"""
calculate_word_avg_price.py
@date: 2020/08/31
"""

import math
import string
import pickle
import enchant
import pandas as pd
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

word_prices = {}
word_count = defaultdict(int)
word_price_avg = {}

for index, df in menu_frame.iterrows():

    if index % 10000 == 0: print ("index:", index)
    item = df["item"]

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
                if word not in word_prices:
                    word_prices[word] = [df["price"]]
                else:
                    word_prices[word].append(df["price"])
                word_count[word] += 1

    except AttributeError:
        pass

for word in word_prices:
    word_price_avg[word] = sum(word_prices[word]) / word_count[word]

pickle.dump(word_price_avg, open("menu_word_avg_price.p", "wb"))
pickle.dump(word_prices, open("menu_word_prices.p", "wb"))
