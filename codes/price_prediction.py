"""
price_prediction.py
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

# model_name = "word2vec"
# dataset = "menu"

for dataset in ["menu", "shoe", "retail", "reward"]:

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

    filtered = False

    for model_name in ["wp", "word2vec", "mwp", "mwcp", "joint_wwm"]:

        X_dict = pickle.load(open("../trained_parameters/"+ model_name + "_" + dataset + "_X.p", "rb"))

        all_words = [word for word in X_dict.keys()]

        if not filtered:

            y = []

            X_raw = []

            for index, df in menu_frame.iterrows():

                try:
                    words = clean(df[item_column_name])
                    price = df[price_column_name]

                    if not math.isnan(price) and all([word in all_words for word in words]):
                        X_raw.append(words)
                        y.append(price)
                except AttributeError:
                    pass

            filtered = True

        X = []

        for words in X_raw:
            tmp = np.zeros(hidden_dim)
            for word in words:
                tmp += X_dict[word]
            X.append(tmp)

        X = np.asarray(X)
        y = np.asarray(y)
        y = y.reshape(-1, 1)

        target_scaler = MinMaxScaler()
        target_scaler.fit(y)
        y = target_scaler.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        lr_model = LinearRegression()
        # wrapped_model = TransformedTargetRegressor(regressor=lr_model, transformer=MinMaxScaler())
        # lr_model = SVR(C=1.0, epsilon=0.2)
        reg = lr_model.fit(X_train, y_train)

        y_pred = lr_model.predict(X_test)

        print (dataset, model_name, mean_squared_error(y_test, y_pred))
